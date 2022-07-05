#include <stdlib.h>
#include "types.h"

struct ListNode {
    void* value;
    struct ListNode* next;
};

struct List {
    int size;
    struct ListNode* root;
    struct ListNode* end;
    void (*append)(struct List* restrict, void* restrict);
    void (*clear)(struct List* restrict);
    struct ListNode* (*get) (struct List* restrict, const int);
    void* (*at)(struct List* restrict, const int);
    void (*set)(struct List* restrict, const int, void** restrict);
};

typedef struct {
    int size;
} init_args;

void append(struct List* restrict self, void* restrict value);
void _destroy(struct ListNode* restrict node);
void clear(struct List* restrict self);
struct ListNode* get(struct List* restrict self, const int index);
void* at(struct List* restrict self, const int index);
void set(struct List* restrict self, const int index, void** restrict value);
struct List* create_list_base(const int size);
struct List* var_init(init_args args);

#define create_list(...) var_init((init_args){__VA_ARGS__});