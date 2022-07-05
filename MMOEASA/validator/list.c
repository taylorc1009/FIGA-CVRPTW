#include "list.h"

struct List* var_init(init_args args) {
    int size_out = args.size? args.size : 0;
    return create_list_base(size_out);
}

struct List* create_list_base(const int size) {
    struct List* list = (struct List*)malloc(sizeof(struct List));
    list->append = append;
    list->clear = clear;
    list->get = get;
    list->at = at;
    list->set = set;

    if (size >= 1) {
        struct ListNode* node = list->root = (struct ListNode*)malloc(sizeof(struct ListNode));

        for(int i = 1; i < size; i++)
            node = node->next = (struct ListNode*)malloc(sizeof(struct ListNode));
        node->next = NULL;
        
        list->end = node;
    }

    list->size = size;

    return list;
}

void append(struct List* restrict self, void* restrict value) {
    if (self->size > 0)
        self->end = self->end->next = (struct ListNode*)malloc(sizeof(struct ListNode));
    else
        self->end = self->root = (struct ListNode*)malloc(sizeof(struct ListNode));
    
    self->end->value = value;
    self->end->next = NULL;

    self->size++;
}

void _destroy(struct ListNode* restrict node) { //should be a private method of the list; therefore, it's not included in the struct definition
    if (node) {
        _destroy(node->next);
        //TODO is this right: don't free node->value in case it's in use by something else?
        free(node);
    }
}

void clear(struct List* restrict self) {
    _destroy(self->root);
    self->size = 0;
}

struct ListNode* get(struct List* restrict self, const int index) {
    if (!(0 <= index < self->size))
        return NULL;

    struct ListNode* node = self->root;
    for (int i = 1; i <= index; i++) //start at index 1 as the previous line already acquires 0 (the root)
        node = node->next;

    return node;
}

void* at(struct List* restrict self, const int index) {
    struct ListNode* node = self->get(self, index);
    return node ? node->value : NULL;
}

void set(struct List* restrict self, const int index, void** restrict value) {
    struct ListNode* node = self->get(self, index);
    //if (node) //don't check this because if the user gave an incorrect index, it should crash; they should know something went wrong
    node->value = *value;
}