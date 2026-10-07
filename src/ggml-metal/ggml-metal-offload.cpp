#include "ggml-metal-offload.h"
#include "ggml-metal-offload-impl.h"

#include "ggml.h"
#include "ggml-impl.h"

#include "ggml-metal-device.h"
#include "ggml-metal-ops.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#define OFFLOAD_MAX_CALLS 1024
#define OFFLOAD_SLOT_WORDS 8

struct offload_runner {
    std::mutex              mu;
    std::condition_variable cv;

    bool has_job = false;
    bool done    = false;
    bool result  = false;
    bool exit    = false;

    ggml_metal_offload_run_t run  = nullptr;
    void *                   user = nullptr;
    void *                   call = nullptr;
};

struct offload_pending {
    uint64_t seq;
    void *   call;
};

struct offload_scratch {
    int64_t K;
    int64_t N;
    void *  buf;
};

struct offload_registration {
    void *                     user;
    int                        timeout_ms;
    ggml_metal_offload_match_t match;
    ggml_metal_offload_run_t   run;

    std::mutex mu;

    bool                 bound       = false;
    const char *         unavailable_reason = nullptr;
    ggml_metal_device_t  dev         = nullptr;
    void *               mtl_dev     = nullptr;
    void *               event       = nullptr;
    void *               fence_buf   = nullptr;
    void *               slot_buf    = nullptr;
    uint32_t *           fence_word  = nullptr;
    uint32_t *           slots       = nullptr;

    uint64_t next_seq = 1;
    uint64_t acc_seq  = 0;

    std::unordered_map<std::string, offload_scratch> scratch;
    std::unordered_set<std::string>                  logged;

    std::vector<std::pair<const ggml_tensor *, ggml_metal_offload_call>> table;

    void * last_cb = nullptr;

    std::thread             poller;
    std::mutex              qmu;
    std::condition_variable qcv;
    std::deque<offload_pending> queue;
    std::atomic<bool>       stop{false};
    std::atomic<uint64_t>   epoch{0};

    std::shared_ptr<offload_runner> runner;

    std::atomic<bool> off{false};
    std::atomic<bool> closing{false};

    int64_t served     = 0;
    int64_t recomputed = 0;
};

static std::mutex                                g_mu;
static std::atomic<offload_registration *>       g_cur{nullptr};

static std::shared_ptr<offload_registration> & g_reg() {
    static auto * reg = new std::shared_ptr<offload_registration>();
    return *reg;
}

static void offload_log_once(offload_registration * r, const std::string & key, const std::string & msg) {
    if (r->logged.insert(key).second) {
        GGML_LOG_WARN("%s: offload: %s\n", __func__, msg.c_str());
    }
}

static void offload_runner_main(std::shared_ptr<offload_runner> st) {
    std::unique_lock<std::mutex> lock(st->mu);

    for (;;) {
        st->cv.wait(lock, [&] { return st->exit || st->has_job; });
        if (st->exit) {
            return;
        }

        st->has_job = false;

        ggml_metal_offload_run_t run  = st->run;
        void *                   user = st->user;
        void *                   call = st->call;

        lock.unlock();
        const bool res = run(user, call);
        lock.lock();

        st->result = res;
        st->done   = true;
        st->cv.notify_all();
    }
}

static bool offload_call_agent(offload_registration * r, void * call) {
    std::shared_ptr<offload_runner> st = r->runner;

    {
        std::lock_guard<std::mutex> lock(st->mu);
        st->call    = call;
        st->done    = false;
        st->has_job = true;
    }
    st->cv.notify_all();

    std::unique_lock<std::mutex> lock(st->mu);

    const bool done = st->cv.wait_for(lock, std::chrono::milliseconds(r->timeout_ms), [&] { return st->done; });
    if (!done) {
        st->exit = true;
        lock.unlock();
        st->cv.notify_all();
        return false;
    }

    return st->result;
}

static bool offload_fence_reached(offload_registration * r, uint64_t seq) {
    const uint32_t v = __atomic_load_n(r->fence_word, __ATOMIC_ACQUIRE);

    return (int32_t) (v - (uint32_t) seq) >= 0;
}

static void offload_decide(offload_registration * r, uint64_t seq, bool ok) {
    uint32_t * slot = r->slots + (seq % OFFLOAD_MAX_CALLS)*OFFLOAD_SLOT_WORDS;

    __atomic_store_n(slot, ok ? 1u : 2u, __ATOMIC_RELEASE);

    ggml_metal_shared_event_set(r->event, seq);
}

static void offload_poller_main(offload_registration * r) {
    for (;;) {
        offload_pending c;
        uint64_t        ep;

        {
            std::unique_lock<std::mutex> lock(r->qmu);

            r->qcv.wait(lock, [&] { return r->stop || !r->queue.empty(); });
            if (r->stop) {
                return;
            }

            c  = r->queue.front();
            ep = r->epoch.load();
        }

        bool ok      = false;
        bool dropped = false;

        if (!r->off.load() && !r->closing.load()) {
            int spins = 0;
            for (;;) {
                if (offload_fence_reached(r, c.seq)) {
                    ok = true;
                    break;
                }
                if (r->off.load() || r->closing.load()) {
                    break;
                }
                if (r->epoch.load() != ep || r->stop.load()) {
                    dropped = true;
                    break;
                }
                if (++spins > 200) {
                    std::this_thread::sleep_for(std::chrono::microseconds(20));
                }
            }
        }

        if (ok && (r->epoch.load() != ep || r->stop.load())) {
            dropped = true;
        }

        if (dropped) {
            continue;
        }

        if (ok && !r->off.load() && !r->closing.load()) {
            ok = offload_call_agent(r, c.call);
            if (!ok) {
                r->off.store(true);
            }
        } else {
            ok = false;
        }

        {
            std::lock_guard<std::mutex> lock(r->qmu);
            if (r->epoch.load() != ep) {
                continue;
            }
            r->queue.pop_front();
        }

        offload_decide(r, c.seq, ok);
    }
}

static const char * offload_bind(offload_registration * r, ggml_metal_device_t dev) {
    if (r->bound) {
        return r->mtl_dev != ggml_metal_device_get_obj(dev) ? "the offload is bound to another device" : nullptr;
    }

    if (r->unavailable_reason) {
        return r->unavailable_reason;
    }

    ggml_metal_library_t lib = ggml_metal_device_get_library(dev);
    if (!lib || !ggml_metal_library_has_function(lib, "kernel_offload_fence")) {
        r->unavailable_reason = "offload kernels unavailable";
        return r->unavailable_reason;
    }

    r->dev       = dev;
    r->mtl_dev   = ggml_metal_device_get_obj(dev);
    r->event     = ggml_metal_device_new_shared_event(dev);
    r->fence_buf = ggml_metal_device_new_shared_buffer(dev, 64);
    r->slot_buf  = ggml_metal_device_new_shared_buffer(dev, OFFLOAD_MAX_CALLS*OFFLOAD_SLOT_WORDS*sizeof(uint32_t));

    if (!r->event || !r->fence_buf || !r->slot_buf) {
        if (r->event)     ggml_metal_object_release(r->event);
        if (r->fence_buf) ggml_metal_object_release(r->fence_buf);
        if (r->slot_buf)  ggml_metal_object_release(r->slot_buf);
        r->event = r->fence_buf = r->slot_buf = nullptr;
        r->unavailable_reason = "cannot allocate the offload buffers";
        return r->unavailable_reason;
    }

    r->fence_word = (uint32_t *) ggml_metal_buffer_contents(r->fence_buf);
    r->slots      = (uint32_t *) ggml_metal_buffer_contents(r->slot_buf);

    memset(r->fence_word, 0, 64);
    memset(r->slots, 0, OFFLOAD_MAX_CALLS*OFFLOAD_SLOT_WORDS*sizeof(uint32_t));

    r->runner       = std::make_shared<offload_runner>();
    r->runner->run  = r->run;
    r->runner->user = r->user;
    std::thread(offload_runner_main, r->runner).detach();

    r->poller = std::thread(offload_poller_main, r);

    r->bound = true;

    return nullptr;
}

static void offload_account(offload_registration * r) {
    for (uint64_t s = r->acc_seq + 1; s < r->next_seq; ++s) {
        const uint32_t * slot = r->slots + (s % OFFLOAD_MAX_CALLS)*OFFLOAD_SLOT_WORDS;

        const uint32_t status = __atomic_load_n(slot, __ATOMIC_ACQUIRE);
        if (status == 0) {
            continue;
        }

        if (status == 2 || slot[1] != 0) {
            r->recomputed++;
        } else {
            r->served++;
        }
    }

    r->acc_seq = r->next_seq - 1;
}

static void offload_finish_previous(offload_registration * r) {
    if (r->last_cb) {
        ggml_metal_cmd_buf_wait_completed(r->last_cb);
        ggml_metal_object_release(r->last_cb);
        r->last_cb = nullptr;
    }

    if (r->bound) {
        offload_account(r);
    }
}

static const char * offload_check_plan(offload_registration * r, const ggml_metal_offload_plan & p, int64_t K, int64_t N, int64_t M) {
    const int64_t G = p.gpu_rows;

    if (G <= 0 || G >= N || G % 64 != 0 || N % 64 != 0) {
        return "gpu rows must be a multiple of 64 inside (0, N)";
    }

    int e = 0;
    if (!(p.in_scale > 0.0f) || std::frexp(p.in_scale, &e) != 0.5f || e - 1 < -8 || e - 1 > 8) {
        return "the input scale must be a power of two from 2^-8 to 2^8";
    }

    if (p.n_seg < 0 || p.n_seg > GGML_METAL_OFFLOAD_MAX_SEGMENTS) {
        return "at most 4 segments";
    }

    if (p.n_seg > 0) {
        int64_t sum = 0;
        for (int i = 0; i < p.n_seg; ++i) {
            if (p.seg[i] <= 0) {
                return "segments must be positive";
            }
            sum += p.seg[i];
        }
        if (sum != M) {
            return "the segments must sum to the input rows";
        }
    }

    if (p.in_stride < (size_t) (2*K) || p.in_stride % 16 != 0 || p.out_stride < (size_t) (2*(N - G)) || p.out_stride % 16 != 0) {
        return "strides must cover a row and be multiples of 16";
    }

    if (!ggml_metal_buffer_fits(r->dev, p.in, (size_t) (M - 1)*p.in_stride + 2*K) ||
        !ggml_metal_buffer_fits(r->dev, p.out, (size_t) (M - 1)*p.out_stride + 2*(N - G))) {
        return "the buffers must be shared buffers of the backend's device and large enough";
    }

    return nullptr;
}

void ggml_metal_offload_prepare(ggml_metal_device_t dev, ggml_cgraph * gf) {
    offload_registration * r = g_cur.load();
    if (!r) {
        return;
    }

    std::lock_guard<std::mutex> lock(r->mu);

    r->table.clear();

    offload_finish_previous(r);

    {
        std::lock_guard<std::mutex> qlock(r->qmu);
        r->queue.clear();
        r->epoch++;
    }

    if (r->off.load()) {
        return;
    }

    const char * bind_reason = offload_bind(r, dev);

    const ggml_metal_device_props * props = ggml_metal_device_get_props(dev);

    std::vector<offload_pending> pend;

    for (int i = 0; i < gf->n_nodes; ++i) {
        ggml_tensor * node = gf->nodes[i];

        if (ggml_op_is_empty(node->op) || ggml_is_empty(node) || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        if (node->op != GGML_OP_MUL_MAT || !ggml_metal_op_mul_mat_use_deq(node, props)) {
            continue;
        }

        const ggml_tensor * a = node->src[0];
        const ggml_tensor * b = node->src[1];

        if (a->ne[2] != 1 || a->ne[3] != 1 || b->ne[2] != 1 || b->ne[3] != 1 || !ggml_is_contiguous(b) || !ggml_is_contiguous(node)) {
            continue;
        }

        const int64_t K = a->ne[0];
        const int64_t N = a->ne[1];
        const int64_t M = b->ne[1];

        ggml_metal_offload_plan p = {};

        if (!r->match(r->user, a->name, K, N, M, &p)) {
            continue;
        }

        const char * reason = bind_reason ? bind_reason : offload_check_plan(r, p, K, N, M);
        if (!reason && pend.size() >= OFFLOAD_MAX_CALLS) {
            reason = "a graph offloads at most 1024 calls";
        }
        if (reason) {
            offload_log_once(r, std::string("plan:") + a->name, std::string(a->name) + ": " + reason);
            continue;
        }

        auto sc = r->scratch.find(a->name);
        if (sc == r->scratch.end()) {
            void * buf = ggml_metal_device_new_shared_buffer(dev, 16*(size_t) (K + N));
            if (!buf) {
                offload_log_once(r, std::string("scratch:") + a->name, std::string(a->name) + ": cannot allocate the scratch buffer");
                continue;
            }
            sc = r->scratch.emplace(a->name, offload_scratch{K, N, buf}).first;
        } else if (sc->second.K != K || sc->second.N != N) {
            offload_log_once(r, std::string("shape:") + a->name, std::string(a->name) + ": the weight changed shape");
            continue;
        }

        const uint64_t seq = r->next_seq++;

        uint32_t * slot = r->slots + (seq % OFFLOAD_MAX_CALLS)*OFFLOAD_SLOT_WORDS;
        memset(slot, 0, OFFLOAD_SLOT_WORDS*sizeof(uint32_t));

        ggml_metal_offload_call c = {};
        c.gpu_rows   = p.gpu_rows;
        c.scale      = p.in_scale;
        c.n_seg      = p.n_seg;
        c.seq        = seq;
        c.in_stride  = p.in_stride;
        c.out_stride = p.out_stride;
        c.in         = { p.in,  0 };
        c.out        = { p.out, 0 };
        c.scratch    = { sc->second.buf, 0 };
        c.fence      = { r->fence_buf, 0 };
        c.slot       = { r->slot_buf, (seq % OFFLOAD_MAX_CALLS)*OFFLOAD_SLOT_WORDS*sizeof(uint32_t) };
        c.event      = r->event;

        int64_t end = 0;
        for (int s = 0; s < p.n_seg; ++s) {
            end += p.seg[s];
            c.seg_end[s] = (int32_t) end;
        }

        r->table.emplace_back(node, c);
        pend.push_back({ seq, p.call });
    }

    if (!pend.empty()) {
        {
            std::lock_guard<std::mutex> qlock(r->qmu);
            for (const offload_pending & c : pend) {
                r->queue.push_back(c);
            }
        }
        r->qcv.notify_all();
    }
}

void ggml_metal_offload_finish(ggml_metal_cmd_buf_t cmd_buf_last) {
    offload_registration * r = g_cur.load();
    if (!r || !cmd_buf_last) {
        return;
    }

    std::lock_guard<std::mutex> lock(r->mu);

    if (r->table.empty()) {
        return;
    }

    if (r->last_cb) {
        ggml_metal_object_release(r->last_cb);
    }
    r->last_cb = ggml_metal_object_retain(cmd_buf_last);
}

const ggml_metal_offload_call * ggml_metal_offload_find(const ggml_tensor * node) {
    offload_registration * r = g_cur.load();
    if (!r) {
        return nullptr;
    }

    for (const auto & e : r->table) {
        if (e.first == node) {
            return &e.second;
        }
    }

    return nullptr;
}

static void offload_teardown(std::shared_ptr<offload_registration> r) {
    r->closing.store(true);

    {
        std::lock_guard<std::mutex> lock(r->mu);

        if (r->last_cb) {
            ggml_metal_cmd_buf_wait_completed(r->last_cb);
            ggml_metal_object_release(r->last_cb);
            r->last_cb = nullptr;
        }
    }

    {
        std::lock_guard<std::mutex> lock(r->qmu);
        r->stop = true;
    }
    r->qcv.notify_all();

    if (r->poller.joinable()) {
        r->poller.join();
    }

    if (r->runner) {
        {
            std::lock_guard<std::mutex> lock(r->runner->mu);
            r->runner->exit = true;
        }
        r->runner->cv.notify_all();
    }

    for (auto & s : r->scratch) {
        ggml_metal_object_release(s.second.buf);
    }
    r->scratch.clear();

    if (r->event)     ggml_metal_object_release(r->event);
    if (r->fence_buf) ggml_metal_object_release(r->fence_buf);
    if (r->slot_buf)  ggml_metal_object_release(r->slot_buf);
}

void ggml_metal_offload_set(void * user, int timeout_ms, ggml_metal_offload_match_t match, ggml_metal_offload_run_t run) {
    GGML_ASSERT(match && run && timeout_ms > 0);

    ggml_metal_offload_remove();

    std::lock_guard<std::mutex> lock(g_mu);

    auto r = std::make_shared<offload_registration>();
    r->user       = user;
    r->timeout_ms = timeout_ms;
    r->match      = match;
    r->run        = run;

    g_reg() = r;
    g_cur.store(r.get());
}

void ggml_metal_offload_remove(void) {
    std::lock_guard<std::mutex> lock(g_mu);

    std::shared_ptr<offload_registration> r = g_reg();
    if (!r) {
        return;
    }

    g_cur.store(nullptr);

    offload_teardown(r);

    g_reg().reset();
}

void ggml_metal_offload_stats(int64_t * served, int64_t * recomputed, bool * off) {
    std::lock_guard<std::mutex> lock(g_mu);

    offload_registration * r = g_cur.load();
    if (!r) {
        *served     = 0;
        *recomputed = 0;
        *off        = false;
        return;
    }

    {
        std::lock_guard<std::mutex> rlock(r->mu);

        offload_finish_previous(r);

        *served     = r->served;
        *recomputed = r->recomputed;
    }

    *off = r->off.load();
}
