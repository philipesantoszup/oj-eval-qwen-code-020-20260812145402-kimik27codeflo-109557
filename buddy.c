#include "buddy.h"

#define NULL ((void *)0)
#define PAGE_SIZE 4096
#define MAX_RANK 16
#define MAX_PAGES 32768

struct FreeBlock {
    struct FreeBlock *next;
    struct FreeBlock *prev;
};

static void *pool_start = NULL;
static int pool_pgcount = 0;
static int pool_max_rank = 0;

static struct FreeBlock *free_list[MAX_RANK + 1];
static unsigned char page_rank_arr[MAX_PAGES];
static unsigned char is_alloc_start[MAX_PAGES];

static inline int compute_max_rank(int pgcount) {
    int r = 1;
    while ((1 << (r - 1)) <= pgcount && r <= MAX_RANK) {
        r++;
    }
    return r - 1;
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

int init_page(void *p, int pgcount) {
    if (p == NULL || pgcount <= 0) return -EINVAL;

    pool_start = p;
    pool_pgcount = pgcount;
    pool_max_rank = compute_max_rank(pgcount);

    for (int i = 0; i < MAX_PAGES; i++) {
        page_rank_arr[i] = 0;
        is_alloc_start[i] = 0;
    }
    for (int i = 1; i <= MAX_RANK; i++) {
        free_list[i] = NULL;
    }

    if (pool_max_rank >= 1) {
        int block_pages = 1 << (pool_max_rank - 1);
        if (block_pages > MAX_PAGES) block_pages = MAX_PAGES;
        if (block_pages > pgcount) block_pages = pgcount;

        struct FreeBlock *block = (struct FreeBlock *)p;
        block->next = NULL;
        block->prev = NULL;
        free_list[pool_max_rank] = block;

        for (int i = 0; i < block_pages; i++) {
            page_rank_arr[i] = (unsigned char)pool_max_rank;
        }
    }

    return OK;
}

void *alloc_pages(int rank) {
    if (rank < 1 || rank > MAX_RANK) {
        return ERR_PTR(-EINVAL);
    }

    int k = rank;
    while (k <= MAX_RANK && free_list[k] == NULL) {
        k++;
    }
    if (k > MAX_RANK) {
        return ERR_PTR(-ENOSPC);
    }

    while (k > rank) {
        struct FreeBlock *block = free_list[k];
        free_list[k] = block->next;
        if (block->next) block->next->prev = NULL;

        int idx = ptr_to_idx(block);
        int half_pages = 1 << (k - 2);
        int right_idx = idx + half_pages;

        struct FreeBlock *right = (struct FreeBlock *)(pool_start + right_idx * PAGE_SIZE);
        right->next = free_list[k - 1];
        right->prev = NULL;
        if (free_list[k - 1]) free_list[k - 1]->prev = right;
        free_list[k - 1] = right;

        struct FreeBlock *left = (struct FreeBlock *)(pool_start + idx * PAGE_SIZE);
        left->next = free_list[k - 1];
        left->prev = NULL;
        if (free_list[k - 1]) free_list[k - 1]->prev = left;
        free_list[k - 1] = left;

        for (int i = idx; i < idx + half_pages; i++) {
            page_rank_arr[i] = (unsigned char)(k - 1);
        }
        for (int i = right_idx; i < right_idx + half_pages; i++) {
            page_rank_arr[i] = (unsigned char)(k - 1);
        }

        k--;
    }

    struct FreeBlock *block = free_list[rank];
    free_list[rank] = block->next;
    if (block->next) block->next->prev = NULL;

    int idx = ptr_to_idx(block);
    is_alloc_start[idx] = 1;

    return (void *)block;
}

int return_pages(void *p) {
    if (!is_valid_ptr(p)) return -EINVAL;

    int idx = ptr_to_idx(p);
    if (!is_alloc_start[idx]) return -EINVAL;

    int r = page_rank_arr[idx];
    is_alloc_start[idx] = 0;

    while (r <= MAX_RANK) {
        int buddy_idx = idx ^ (1 << (r - 1));
        if (buddy_idx < 0 || buddy_idx >= pool_pgcount) break;
        if (buddy_idx + (1 << (r - 1)) > pool_pgcount) break;
        if (is_alloc_start[buddy_idx]) break;
        if (page_rank_arr[buddy_idx] != r) break;

        struct FreeBlock *buddy = (struct FreeBlock *)(pool_start + buddy_idx * PAGE_SIZE);
        if (buddy->prev) buddy->prev->next = buddy->next;
        else free_list[r] = buddy->next;
        if (buddy->next) buddy->next->prev = buddy->prev;

        if (buddy_idx < idx) idx = buddy_idx;
        r++;

        int block_pages = 1 << (r - 1);
        for (int i = idx; i < idx + block_pages; i++) {
            page_rank_arr[i] = (unsigned char)r;
        }
    }

    struct FreeBlock *block = (struct FreeBlock *)(pool_start + idx * PAGE_SIZE);
    block->next = free_list[r];
    block->prev = NULL;
    if (free_list[r]) free_list[r]->prev = block;
    free_list[r] = block;

    return OK;
}

int query_ranks(void *p) {
    if (!is_valid_ptr(p)) return -EINVAL;
    int idx = ptr_to_idx(p);
    int r = page_rank_arr[idx];
    if (r == 0) return -EINVAL;
    return r;
}

int query_page_counts(int rank) {
    if (rank < 1 || rank > MAX_RANK) return -EINVAL;
    int count = 0;
    struct FreeBlock *b = free_list[rank];
    while (b) {
        count++;
        b = b->next;
    }
    return count;
}
