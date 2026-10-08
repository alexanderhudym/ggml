#include "ggml-metal-offload.h"
#include "ggml-metal-split.h"

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-metal-device.h"
#include "ggml-metal-ops.h"

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#define SPLIT_SLOTS 2

struct split_entry {
    std::string name;
    ggml_type   type;
    int64_t     ne0;
    int64_t     ne1;
    size_t      gpu_bytes;
    size_t      tail_bytes;
};

struct split_record {
    const void * ctx;
    int32_t      index;
};

struct split_job {
    uint64_t fseq;
    int32_t  index;
};

struct split_state {
    void *                  user;
    ggml_metal_split_read_t read;

    std::vector<split_entry>               entries;
    std::unordered_map<std::string, int32_t> by_name;

    size_t page      = 0;
    size_t slot_size = 0;
    size_t host_size = 0;
    char * host      = nullptr;

    std::shared_mutex                              rmu;
    std::unordered_map<const void *, split_record> records;
    std::atomic<size_t>                            n_records{0};

    std::mutex mu;

    bool                bound     = false;
    ggml_metal_device_t dev       = nullptr;
    void *              mtl_dev   = nullptr;
    void *              region    = nullptr;
    void *              event     = nullptr;
    uint32_t *          fence_word = nullptr;

    uint64_t next_fseq = 1;
    void *   last_cb   = nullptr;

    std::unordered_map<const ggml_tensor *, ggml_metal_split_call> table;

    std::thread                thread;
    std::mutex                 qmu;
    std::condition_variable    qcv;
    std::deque<split_job>      queue;
    std::atomic<bool>          stop{false};
    std::atomic<uint64_t>      epoch{0};

    std::atomic<int64_t> reads{0};
    std::atomic<bool>    failed{false};
};

static std::mutex                                g_mu;
static std::atomic<split_state *>                g_cur{nullptr};

static std::shared_ptr<split_state> & g_reg() {
    static auto * reg = new std::shared_ptr<split_state>();
    return *reg;
}

static bool split_fence_reached(split_state * s, uint64_t seq) {
    const uint32_t v = __atomic_load_n(s->fence_word, __ATOMIC_ACQUIRE);

    return (int32_t) (v - (uint32_t) seq) >= 0;
}

static char * split_slot_ptr(split_state * s, uint64_t fseq) {
    return s->host + s->page + (fseq % SPLIT_SLOTS)*s->slot_size;
}

static void split_thread_main(split_state * s) {
    for (;;) {
        split_job j;
        uint64_t  ep;

        {
            std::unique_lock<std::mutex> lock(s->qmu);

            s->qcv.wait(lock, [&] { return s->stop || !s->queue.empty(); });
            if (s->stop) {
                return;
            }

            j  = s->queue.front();
            ep = s->epoch.load();
        }

        bool dropped = false;
        int  spins   = 0;

        while (!split_fence_reached(s, j.fseq - SPLIT_SLOTS)) {
            if (s->epoch.load() != ep || s->stop.load()) {
                dropped = true;
                break;
            }
            if (++spins > 200) {
                std::this_thread::sleep_for(std::chrono::microseconds(20));
            }
        }

        if (dropped) {
            continue;
        }

        const split_entry & e = s->entries[j.index];

        s->reads++;
        if (!s->read(s->user, j.index, 0, split_slot_ptr(s, j.fseq), e.tail_bytes)) {
            s->failed.store(true);
        }

        {
            std::lock_guard<std::mutex> lock(s->qmu);
            if (s->epoch.load() != ep) {
                continue;
            }
            s->queue.pop_front();
        }

        ggml_metal_shared_event_set(s->event, j.fseq);
    }
}

static void split_bind(split_state * s, ggml_metal_device_t dev) {
    if (s->bound) {
        GGML_ASSERT(s->mtl_dev == ggml_metal_device_get_obj(dev) && "the split weights are bound to another device");
        return;
    }

    ggml_metal_library_t lib = ggml_metal_device_get_library(dev);
    GGML_ASSERT(lib && ggml_metal_library_has_function(lib, "kernel_offload_fence") &&
                ggml_metal_library_has_function(lib, "kernel_split_copy") && "split weights need the offload kernels");

    s->dev     = dev;
    s->mtl_dev = ggml_metal_device_get_obj(dev);
    s->region  = ggml_metal_device_wrap_buffer(dev, s->host, s->host_size);
    s->event   = ggml_metal_device_new_shared_event(dev);

    GGML_ASSERT(s->region && s->event && "cannot bind the split weight slots to the device");

    s->fence_word = (uint32_t *) s->host;
    s->bound      = true;
}

static void split_finish_previous(split_state * s) {
    if (s->last_cb) {
        ggml_metal_cmd_buf_wait_completed(s->last_cb);
        ggml_metal_object_release(s->last_cb);
        s->last_cb = nullptr;
    }
}

static int32_t split_find_entry(const split_state * s, const ggml_tensor * t) {
    if (t->op != GGML_OP_NONE || t->view_src || t->ne[2] != 1 || t->ne[3] != 1) {
        return -1;
    }

    auto it = s->by_name.find(t->name);
    if (it == s->by_name.end()) {
        return -1;
    }

    const split_entry & e = s->entries[it->second];

    return e.type == t->type && e.ne0 == t->ne[0] && e.ne1 == t->ne[1] ? it->second : -1;
}

bool ggml_metal_split_active(void) {
    split_state * s = g_cur.load();

    return s && s->n_records.load() > 0;
}

size_t ggml_metal_split_alloc_size(const ggml_tensor * t) {
    split_state * s = g_cur.load();
    if (!s) {
        return 0;
    }

    const int32_t i = split_find_entry(s, t);

    return i < 0 ? 0 : s->entries[i].gpu_bytes;
}

bool ggml_metal_split_matches(const ggml_tensor * t) {
    split_state * s = g_cur.load();

    return s && split_find_entry(s, t) >= 0;
}

void ggml_metal_split_record(void * ctx, const ggml_tensor * t) {
    split_state * s = g_cur.load();
    if (!s) {
        return;
    }

    const int32_t i = split_find_entry(s, t);
    if (i < 0) {
        return;
    }

    std::unique_lock<std::shared_mutex> lock(s->rmu);

    s->records[t->data] = { ctx, i };
    s->n_records.store(s->records.size());
}

void ggml_metal_split_drop(void * ctx) {
    split_state * s = g_cur.load();
    if (!s || s->n_records.load() == 0) {
        return;
    }

    std::unique_lock<std::shared_mutex> lock(s->rmu);

    for (auto it = s->records.begin(); it != s->records.end();) {
        it = it->second.ctx == ctx ? s->records.erase(it) : std::next(it);
    }
    s->n_records.store(s->records.size());
}

bool ggml_metal_split_lookup(const void * ctx, const ggml_tensor * t, size_t * gpu_bytes, int32_t * index, size_t * view_offs) {
    split_state * s = g_cur.load();
    if (!s || s->n_records.load() == 0) {
        return false;
    }

    const ggml_tensor * root = t->view_src ? t->view_src : t;

    std::shared_lock<std::shared_mutex> lock(s->rmu);

    auto it = s->records.find(root->data);
    if (it == s->records.end() || it->second.ctx != ctx) {
        return false;
    }

    if (view_offs) {
        *view_offs = t->view_src ? t->view_offs : 0;
    }

    if (gpu_bytes) {
        *gpu_bytes = s->entries[it->second.index].gpu_bytes;
    }
    if (index) {
        *index = it->second.index;
    }

    return true;
}

bool ggml_metal_split_is_alloc(const ggml_tensor * t) {
    return t && t->buffer && ggml_metal_split_lookup(t->buffer->context, t, nullptr, nullptr, nullptr);
}

size_t ggml_metal_split_extent(const void * ctx, const ggml_tensor * t) {
    size_t gpu_bytes = 0;
    size_t view_offs = 0;

    if (!ggml_metal_split_lookup(ctx, t, &gpu_bytes, nullptr, &view_offs) || gpu_bytes <= view_offs) {
        return 0;
    }

    return std::min(gpu_bytes - view_offs, ggml_nbytes(t));
}

bool ggml_metal_split_read_tail(int32_t index, size_t offset, void * dst, size_t size) {
    split_state * s = g_cur.load();
    GGML_ASSERT(s && index >= 0 && index < (int32_t) s->entries.size());

    if (!s->read(s->user, index, offset, dst, size)) {
        s->failed.store(true);
        memset(dst, 0, size);
        return false;
    }

    return true;
}

void ggml_metal_split_prepare(ggml_metal_device_t dev, ggml_cgraph * gf) {
    split_state * s = g_cur.load();
    if (!s) {
        return;
    }

    std::lock_guard<std::mutex> lock(s->mu);

    s->table.clear();

    std::vector<std::pair<const ggml_tensor *, int32_t>> nodes;

    for (int i = 0; i < gf->n_nodes; ++i) {
        ggml_tensor * node = gf->nodes[i];

        if (ggml_op_is_empty(node->op) || ggml_is_empty(node) || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        int32_t index = 0;
        if (node->op == GGML_OP_MUL_MAT && node->src[0]->buffer && !node->src[0]->view_src &&
            ggml_metal_split_lookup(node->src[0]->buffer->context, node->src[0], nullptr, &index, nullptr)) {
            nodes.emplace_back(node, index);
        }
    }

    if (nodes.empty()) {
        return;
    }

    split_finish_previous(s);
    split_bind(s, dev);

    {
        std::lock_guard<std::mutex> qlock(s->qmu);
        s->queue.clear();
        s->epoch++;
    }

    const uint64_t first = s->next_fseq;

    __atomic_store_n(s->fence_word, (uint32_t) (first - 1), __ATOMIC_RELEASE);

    std::vector<split_job> jobs;

    for (const auto & n : nodes) {
        const split_entry & e = s->entries[n.second];

        const uint64_t fseq = s->next_fseq++;

        ggml_metal_split_call c = {};
        c.fseq       = fseq;
        c.index      = n.second;
        c.gpu_bytes  = e.gpu_bytes;
        c.tail_bytes = e.tail_bytes;
        c.slot       = { s->region, s->page + (fseq % SPLIT_SLOTS)*s->slot_size };
        c.fence      = { s->region, 0 };
        c.event      = s->event;

        s->table.emplace(n.first, c);
        jobs.push_back({ fseq, n.second });
    }

    {
        std::lock_guard<std::mutex> qlock(s->qmu);
        for (const split_job & j : jobs) {
            s->queue.push_back(j);
        }
    }
    s->qcv.notify_all();
}

void ggml_metal_split_finish(ggml_metal_cmd_buf_t cmd_buf_last) {
    split_state * s = g_cur.load();
    if (!s || !cmd_buf_last) {
        return;
    }

    std::lock_guard<std::mutex> lock(s->mu);

    if (s->table.empty()) {
        return;
    }

    if (s->last_cb) {
        ggml_metal_object_release(s->last_cb);
    }
    s->last_cb = ggml_metal_object_retain(cmd_buf_last);
}

const ggml_metal_split_call * ggml_metal_split_find(const ggml_tensor * node) {
    split_state * s = g_cur.load();
    if (!s) {
        return nullptr;
    }

    auto it = s->table.find(node);

    return it == s->table.end() ? nullptr : &it->second;
}

bool ggml_metal_split_set(void * user, const ggml_metal_split_weight * w, int32_t n, ggml_metal_split_read_t read) {
    if (!w || n <= 0 || !read) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_mu);

    GGML_ASSERT(!g_reg() && "the split weights are registered already");

    auto s = std::make_shared<split_state>();
    s->user = user;
    s->read = read;

    size_t max_tail = 0;

    for (int32_t i = 0; i < n; ++i) {
        const ggml_metal_split_weight & x = w[i];

        if (!x.name || !x.name[0] || x.type < 0 || x.type >= GGML_TYPE_COUNT || x.ne0 <= 0 || x.ne1 <= 0) {
            return false;
        }

        if (!ggml_metal_op_mul_mat_deq_type((ggml_type) x.type)) {
            return false;
        }

        const ggml_type_traits * traits = ggml_get_type_traits((ggml_type) x.type);
        if (!traits || traits->blck_size <= 0 || traits->type_size == 0 || x.ne0 % traits->blck_size != 0) {
            return false;
        }

        if (x.gpu_rows <= 0 || x.gpu_rows >= x.ne1 || x.gpu_rows % 64 != 0) {
            return false;
        }

        const size_t nb1 = ggml_row_size((ggml_type) x.type, x.ne0);

        split_entry e;
        e.name       = x.name;
        e.type       = (ggml_type) x.type;
        e.ne0        = x.ne0;
        e.ne1        = x.ne1;
        e.gpu_bytes  = (size_t) x.gpu_rows*nb1;
        e.tail_bytes = (size_t) (x.ne1 - x.gpu_rows)*nb1;

        if (!s->by_name.emplace(e.name, (int32_t) s->entries.size()).second) {
            return false;
        }

        max_tail = std::max(max_tail, e.tail_bytes);
        s->entries.push_back(std::move(e));
    }

    s->page      = (size_t) sysconf(_SC_PAGESIZE);
    s->slot_size = (max_tail + s->page - 1)/s->page*s->page;
    s->host_size = s->page + SPLIT_SLOTS*s->slot_size;

    void * host = nullptr;
    if (posix_memalign(&host, s->page, s->host_size) != 0) {
        return false;
    }

    s->host = (char *) host;
    memset(s->host, 0, s->page);

    s->thread = std::thread(split_thread_main, s.get());

    g_reg() = s;
    g_cur.store(s.get());

    return true;
}

void ggml_metal_split_remove(void) {
    std::lock_guard<std::mutex> lock(g_mu);

    std::shared_ptr<split_state> s = g_reg();
    if (!s) {
        return;
    }

    GGML_ASSERT(s->n_records.load() == 0 && "split weights are still allocated");

    g_cur.store(nullptr);

    {
        std::lock_guard<std::mutex> slock(s->mu);
        split_finish_previous(s.get());
    }

    {
        std::lock_guard<std::mutex> qlock(s->qmu);
        s->stop = true;
    }
    s->qcv.notify_all();

    if (s->thread.joinable()) {
        s->thread.join();
    }

    if (s->event)  ggml_metal_object_release(s->event);
    if (s->region) ggml_metal_object_release(s->region);

    free(s->host);

    g_reg().reset();
}

void ggml_metal_split_stats(int32_t * tensors, size_t * dropped, int64_t * reads, bool * failed) {
    std::lock_guard<std::mutex> lock(g_mu);

    split_state * s = g_cur.load();

    *tensors = 0;
    *dropped = 0;
    *reads   = 0;
    *failed  = false;

    if (!s) {
        return;
    }

    {
        std::shared_lock<std::shared_mutex> rlock(s->rmu);

        for (const auto & r : s->records) {
            *tensors += 1;
            *dropped += s->entries[r.second.index].tail_bytes;
        }
    }

    *reads  = s->reads.load();
    *failed = s->failed.load();
}
