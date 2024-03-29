#ifndef PROCESS_SIMU_H
#define PROCESS_SIMU_H

#include <stddef.h>
#include <stdint.h>


typedef struct pcb      pcb_t;
typedef struct pcb_node pcb_node_t;

typedef struct process_status {
    uint32_t    eax;
    uint32_t    ebx;
    uint32_t    ecx;
    uint32_t    edx;
    uint32_t    pc;
    uint16_t    psw;
    void*       user_stack;
} process_status_t;

typedef struct process_dispath_info {
    uint8_t     status;
    uint8_t     priority;
    uint32_t    wait_count;
    uint32_t    live_count;
    uint32_t    event;
} process_dispath_info_t;

typedef struct process_control_info {
    uint32_t    inner_addr;
    uint32_t    outer_addr;
    void*       resource_list;
    pcb_t*      next;
} process_control_info_t;

struct pcb {
    int32_t    pid;
    int32_t    ppid;
    int32_t    uid;

    process_status_t         status;
    process_dispath_info_t   dispath_info;
    process_control_info_t   control_info;

    pcb_node_t* node;
};

typedef struct pcb_list {
    pcb_t*      first;
    pcb_t*      last;
    size_t      size;
} pcb_list_t;

struct pcb_node {
    pcb_t*              pcb;
    struct pcb_node*    child;
    struct pcb_node*    sibling;
};

typedef struct pcb_tree {
    pcb_node_t*         root;
} pcb_tree_t;

int32_t fork(int32_t ppid);

pcb_list_t* get_pcb_list();
pcb_tree_t* get_pcb_tree();

void terminate_all();

#endif // process_simu.h
