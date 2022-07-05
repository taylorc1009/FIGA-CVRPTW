#include "data.h"

struct MMOEASASolution* var_io(io_args in) {
    char* file_out = in.file ? in.file : "solution.csv";
    return read_csv_base(file_out);
}

bool inline is_num_end(const int c) {
    return c == ',' || c == '\n' || c == EOF;
}

char* read_num(FILE* restrict file) {
    size_t size = 10;
    char* str = (char*)malloc(sizeof(char) * size);
    int c, len = 0;

    while (!is_num_end(c = getc(file))) {
        str[len++] = c;

        if (len == size - 1) {
            str = realloc(str, sizeof(*str) * (size += 10));

            if(!str)
                return str;
        }
    }

    str[len++] = '\0';
    str = realloc(str, sizeof(*str) * len);
    return str;
}

void inline read_int(FILE* restrict file, int* restrict var) {
    char* str = read_num(file);
    *var = strtol(str, NULL, 10);
    free(str);
}

void inline read_float(FILE* restrict file, float* restrict var) {
    char* str = read_num(file);
    *var = strtof(str, NULL);
    free(str);
}

struct MMOEASASolution* read_csv_base(char* restrict filename) {
    FILE* file;
    struct MMOEASASolution* solution = NULL;

    if (file = fopen(filename, "rb")) {
        fseek(file, 0, SEEK_END);
        long len = (long)ftell(file);
        if (len > 0) {
            rewind(file);

            solution = (struct MMOEASASolution*)malloc(sizeof(struct MMOEASASolution));

            read_int(file, &solution->vehicle_max_capacity);
            read_int(file, (int*)&solution->feasible);
            read_float(file, &solution->total_distance);
            read_float(file, &solution->distance_unbalance);
            read_float(file, &solution->cargo_unbalance);
            
            int num_vehicles;
            read_int(file, &num_vehicles);
            solution->vehicles = create_list(.size=num_vehicles);

            for (int v = 0; v < num_vehicles; v++) {
                struct Vehicle* vehicle = (struct Vehicle*)malloc(sizeof(struct Vehicle));

                read_float(file, &vehicle->current_capacity);
                read_float(file, &vehicle->route_distance);

                int num_destinations;
                read_int(file, &num_destinations);
                vehicle->destinations = create_list(.size=num_destinations);

                for (int d = 0; d < num_destinations; d++) {
                    struct Destination* destination = (struct Destination*)malloc(sizeof(struct Destination));

                    read_float(file, &destination->arrival_time);
                    read_float(file, &destination->departure_time);
                    read_float(file, &destination->wait_time);

                    struct Node* node = destination->node = (struct Node*)malloc(sizeof(struct Node));

                    read_int(file, &node->number);
                    read_int(file, &node->x);
                    read_int(file, &node->y);
                    read_int(file, &node->demand);
                    read_int(file, &node->ready_time);
                    read_int(file, &node->due_date);
                    read_int(file, &node->service_duration);

                    vehicle->destinations->set(vehicle->destinations, d, (void**)&destination);
                }

                solution->vehicles->set(solution->vehicles, v, (void**)&vehicle);
            }
        }

        fclose(file);
    }

    return solution;
}