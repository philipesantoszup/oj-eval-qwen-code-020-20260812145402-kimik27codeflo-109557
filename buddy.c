#include "buddy.h"

#define NULL ((void *)0)
#define PAGE_SIZE 4096
#define MAX_RANK 16
#define MAX_NODES ((1 << MAX_RANK) - 1)

#define STATE_FREE 0
#define STATE_ALLOCATED 1
#define STATE_SPLIT 2

static void *pool_start = NULL;
static int pool_pgcount = 0;
static int pool_max_rank = 0;

static unsigned char node_state[MAX_NODES + 1];
static unsigned char has_free[MAX_NODES + 1];
static int free_count[MAX_RANK + 1];

static inline int compute_max_rank(int pgcount) {
    int r = 1;
    while (r < MAX_RANK && (1 << (r - 1)) < pgcount) {
        r++;
    }
    return r;
}

static inline void set_has_free(int node) {
    if (node_state[node] == STATE_FREE) {
        has_free[node] = 1;
    } else if (node_state[node] == STATE_ALLOCATED) {
        has_free[node] = 0;
    } else {
        has_free[node] = has_free[node << 1] | has_free[(node << 1) | 1];
    }
}

static void build_tree(int node, int level, int start_page, int pgcount) {
    int size_pages = 1 << (pool_max_rank - 1 - level);
    int end_page = start_page + size_pages;
    if (end_page <= pgcount) {
        node_state[node] = STATE_FREE;
        has_free[node] = 1;
        free_count[pool_max_rank - level]++;
    } else if (start_page >= pgcount) {
        node_state[node] = STATE_ALLOCATED;
        has_free[node] = 0;
    } else {
        node_state[node] = STATE_SPLIT;
        has_free[node] = 1;
        int mid = start_page + (size_pages >> 1);
        build_tree(node << 1, level + 1, start_page, pgcount);
        build_tree((node << 1) | 1, level + 1, mid, pgcount);
        set_has_free(node);
    }
}

static inline int ptr_to_idx(void *p) {
    return (int)(((unsigned long)p - (unsigned long)pool_start) >> 12);
}

static inline int is_valid_ptr(void *p) {
    unsigned long addr = (unsigned long)p;
    unsigned long base = (unsigned long)pool_start;
    if (addr < base) return 0;
    unsigned long diff = addr - base;
    if ((diff & (PAGE_SIZE - 1)) != 0) return 0;
    int idx = (int)(diff >> 12);
    if (idx < 0 || idx >= pool_pgcount) return 0;
    return 1;
}

static inline int node_level(int node) {
    return 31 - __builtin_clz(node);
}

static inline int node_rank(int node) {
    return pool_max_rank - node_level(node);
}

static inline void *node_to_ptr(int node) {
    int level = node_level(node);
    int start_page = (node - (1 << level)) << (pool_max_rank - 1 - level);
    return pool_start + start_page * PAGE_SIZE;
}

static int alloc_recursive(int node, int rank, int target_rank) {
    if (node < 1 || node > MAX_NODES) return 0;
    if (!has_free[node]) return 0;
    if (node_state[node] == STATE_FREE) {
        if (rank == target_rank) {
            node_state[node] = STATE_ALLOCATED;
            set_has_free(node);
            free_count[rank]--;
            return node;
        }
        if (rank < target_rank) return 0;
        node_state[node] = STATE_SPLIT;
        free_count[rank]--;
        int left = node << 1;
        int right = left | 1;
        node_state[left] = STATE_FREE;
        node_state[right] = STATE_FREE;
        set_has_free(left);
        set_has_free(right);
        free_count[rank - 1] += 2;
        int res = alloc_recursive(left, rank - 1, target_rank);
        set_has_free(node);
        return res;
    }
    int left = node << 1;
    int right = left | 1;
    int res = alloc_recursive(left, rank - 1, target_rank);
    if (!res) res = alloc_recursive(right, rank - 1, target_rank);
    set_has_free(node);
    return res;
}

int init_page(void *p, int pgcount) {
    if (p == NULL || pgcount <= 0) return -EINVAL;

    pool_start = p;
    pool_pgcount = pgcount;
    pool_max_rank = compute_max_rank(pgcount);

    for (int i = 1; i <= MAX_NODES; i++) {
        node_state[i] = STATE_ALLOCATED;
        has_free[i] = 0;
    }
    for (int i = 1; i <= MAX_RANK; i++) {
        free_count[i] = 0;
    }

    int managed_pages = 1 << (pool_max_rank - 1);
    int effective_pgcount = pgcount < managed_pages ? pgcount : managed_pages;
    build_tree(1, 0, 0, effective_pgcount);

    return OK;
}

void *alloc_pages(int rank) {
    if (rank < 1 || rank > MAX_RANK) {
        return ERR_PTR(-EINVAL);
    }
    if (pool_max_rank < 1 || !has_free[1]) {
        return ERR_PTR(-ENOSPC);
    }

    int node = alloc_recursive(1, pool_max_rank, rank);
    if (!node) {
        return ERR_PTR(-ENOSPC);
    }
    return node_to_ptr(node);
}

int return_pages(void *p) {
    if (!is_valid_ptr(p)) return -EINVAL;

    int page_idx = ptr_to_idx(p);
    int managed_pages = 1 << (pool_max_rank - 1);
    if (page_idx >= managed_pages) return -EINVAL;

    int node = 1;
    int level = 0;
    while (1) {
        int start_page = (node - (1 << level)) << (pool_max_rank - 1 - level);
        if (node_state[node] == STATE_FREE) return -EINVAL;
        if (node_state[node] == STATE_ALLOCATED) {
            if (start_page != page_idx) return -EINVAL;
            break;
        }
        int size_pages = 1 << (pool_max_rank - 1 - level);
        int mid = start_page + (size_pages >> 1);
        level++;
        if (page_idx < mid) node <<= 1;
        else node = (node << 1) | 1;
    }

    int r = pool_max_rank - level;
    node_state[node] = STATE_FREE;
    free_count[r]++;
    set_has_free(node);

    while (r < pool_max_rank) {
        int parent = node >> 1;
        int left = parent << 1;
        int right = left | 1;
        if (node_state[left] == STATE_FREE && node_state[right] == STATE_FREE) {
            node_state[parent] = STATE_FREE;
            free_count[r] -= 2;
            free_count[r + 1]++;
            node = parent;
            r++;
            set_has_free(node);
        } else {
            break;
        }
    }

    node >>= 1;
    while (node >= 1) {
        set_has_free(node);
        node >>= 1;
    }

    return OK;
}

int query_ranks(void *p) {
    if (!is_valid_ptr(p)) return -EINVAL;

    int page_idx = ptr_to_idx(p);
    int managed_pages = 1 << (pool_max_rank - 1);
    if (page_idx >= managed_pages) return -EINVAL;

    int node = 1;
    int level = 0;
    while (1) {
        if (node_state[node] == STATE_FREE || node_state[node] == STATE_ALLOCATED) {
            return pool_max_rank - level;
        }
        int start_page = (node - (1 << level)) << (pool_max_rank - 1 - level);
        int size_pages = 1 << (pool_max_rank - 1 - level);
        int mid = start_page + (size_pages >> 1);
        level++;
        if (page_idx < mid) node <<= 1;
        else node = (node << 1) | 1;
    }
}

int query_page_counts(int rank) {
    if (rank < 1 || rank > MAX_RANK || rank > pool_max_rank) return -EINVAL;
    return free_count[rank];
}
