
include make.inc

BIN_DIR = bin

.PHONY: all memory traids dmvm dgemm two_scale 

all : memory traids dmvm dgemm two_scale 

MEM_TEST = mem mem_chain2 mem_chain4 mem_chain6 mem_chain8 mem_chain10 mem_chain12 mem_chain14 \
	mem_chain16 mem_address_ordered_traveral mem_random_without_ptrchase \
	mem_seqential_without_ptrchase mem_prefetch mem_chain1_omp mem_chain10_omp \
	mem_random_without_ptrchase_omp mem_seqential_without_ptrchase_omp

TRAIDS_TEST = triads_naive triads_avx512 triads_avx512_nt triads_model

DMVM_TEST = dmvm_naive dmvm_unroll2 dmvm_tiling dmvm_data_traffic_naive dmvm_data_traffic_unroll2 dmvm_data_traffic_tiling 

DGEMM_TEST = dgemm_naive dgemm_recursively_blocked

TWO_SCALE_TEST = two_scale_naive two_scale_fused

MEM_BIN = $(patsubst %, $(BIN_DIR)/%, $(MEM_TEST))
TRAIDS_BIN = $(patsubst %, $(BIN_DIR)/%, $(TRAIDS_TEST))
DMVM_BIN = $(patsubst %, $(BIN_DIR)/%, $(DMVM_TEST))
DGEMM_BIN = $(patsubst %, $(BIN_DIR)/%, $(DGEMM_TEST))
TWO_SCALE_BIN = $(patsubst %, $(BIN_DIR)/%, $(TWO_SCALE_TEST))

ALL_BIN = $(MEM_BIN) $(TRAIDS_BIN) $(DMVM_BIN) $(DGEMM_BIN) $(TWO_SCALE_BIN)

memory : $(MEM_BIN)
traids : $(TRAIDS_BIN)
dmvm : $(DMVM_BIN)
dgemm : $(DGEMM_BIN)
two_scale : $(TWO_SCALE_BIN)


$(MEM_BIN) : % : mem.c mem.h util.h
	$(CC) $(CFLAGS) -o $@ $<

$(TRAIDS_BIN) : % : triads.c util.h
	$(CC) $(CFLAGS) -o $@ $<

$(DMVM_BIN) : % : dmvm.c util.h
	$(CC) $(CFLAGS) -o $@ $<

$(DGEMM_BIN) : % : dgemm.c util.h
	$(CC) $(CFLAGS) -o $@ $<

$(TWO_SCALE_BIN) : % : two_scale.c util.h
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -rf $(ALL_BIN)

$(BIN_DIR)/mem : override CFLAGS += -DDEF_PTRCHASE -DDEF_CHAIN_COUNT=1
$(BIN_DIR)/mem_chain2 : override CFLAGS += -DDEF_PTRCHASE -DDEF_CHAIN_COUNT=2
$(BIN_DIR)/mem_chain4 : override CFLAGS += -DDEF_PTRCHASE -DDEF_CHAIN_COUNT=4
$(BIN_DIR)/mem_chain6 : override CFLAGS += -DDEF_PTRCHASE -DDEF_CHAIN_COUNT=6
$(BIN_DIR)/mem_chain8 : override CFLAGS += -DDEF_PTRCHASE -DDEF_CHAIN_COUNT=8
$(BIN_DIR)/mem_chain10 : override CFLAGS += -DDEF_PTRCHASE -DDEF_CHAIN_COUNT=10
$(BIN_DIR)/mem_chain12 : override CFLAGS += -DDEF_PTRCHASE -DDEF_CHAIN_COUNT=12
$(BIN_DIR)/mem_chain14 : override CFLAGS += -DDEF_PTRCHASE -DDEF_CHAIN_COUNT=14
$(BIN_DIR)/mem_chain16 : override CFLAGS += -DDEF_PTRCHASE -DDEF_CHAIN_COUNT=16
$(BIN_DIR)/mem_address_ordered_traveral : override CFLAGS += -DDEF_PTRCHASE -DDEF_CHAIN_COUNT=1 -DDEF_GEN_SEQUENTIAL_LIST
$(BIN_DIR)/mem_random_without_ptrchase : override CFLAGS += -DDEF_CHAIN_COUNT=1 -DDEF_GEN_RANDOM_LIST -DDEF_RANDOM_WITHOUT_PTRCHASE
$(BIN_DIR)/mem_seqential_without_ptrchase : override CFLAGS += -DDEF_CHAIN_COUNT=1 -DDEF_GEN_SEQUENTIAL_LIST -DDEF_SEQENTIAL_WITHOUT_PTRCHASE
$(BIN_DIR)/mem_prefetch : override CFLAGS += -DDEF_PTRCHASE -DDEF_CHAIN_COUNT=1  -DDEF_GEN_RANDOM_LIST -DDEF_PREFETCH
$(BIN_DIR)/mem_chain1_omp : override CFLAGS += -fopenmp -DDEF_PTRCHASE -DDEF_CHAIN_COUNT=1  -DDEF_GEN_RANDOM_LIST
$(BIN_DIR)/mem_chain10_omp : override CFLAGS += -fopenmp -DDEF_PTRCHASE -DDEF_CHAIN_COUNT=10  -DDEF_GEN_RANDOM_LIST
$(BIN_DIR)/mem_random_without_ptrchase_omp : override CFLAGS += -fopenmp -DDEF_CHAIN_COUNT=1 -DDEF_GEN_RANDOM_LIST -DDEF_RANDOM_WITHOUT_PTRCHASE
$(BIN_DIR)/mem_seqential_without_ptrchase_omp : override CFLAGS += -fopenmp -DDEF_CHAIN_COUNT=1 -DDEF_GEN_SEQUENTIAL_LIST -DDEF_SEQENTIAL_WITHOUT_PTRCHASE

$(BIN_DIR)/triads_naive : override CFLAGS += -DVECTOR_TRIADS_NAIVE
$(BIN_DIR)/triads_avx512 : override CFLAGS += -DVECTOR_TRIADS_AVX512
$(BIN_DIR)/triads_avx512_nt : override CFLAGS += -DVECTOR_TRIADS_AVX512_NT
$(BIN_DIR)/triads_model : override CFLAGS += -DVECTOR_TRIADS_MODEL

$(BIN_DIR)/dmvm_naive : override CFLAGS += -DDMVM_NAIVE
$(BIN_DIR)/dmvm_unroll2 : override CFLAGS += -DDMVM_UNROLL2
$(BIN_DIR)/dmvm_tiling : override CFLAGS += -DDMVM_TILING
$(BIN_DIR)/dmvm_data_traffic_naive : override CFLAGS += -DDMVM_DATA_TRAFFIC_NAIVE
$(BIN_DIR)/dmvm_data_traffic_unroll2 : override CFLAGS += -DDMVM_DATA_TRAFFIC_UNROLL2
$(BIN_DIR)/dmvm_data_traffic_tiling : override CFLAGS += -DDMVM_DATA_TRAFFIC_TILING

$(BIN_DIR)/dgemm_naive : override CFLAGS += -DDGEMM_NAIVE $(PAPI_INC) $(PAPI_LIB)
$(BIN_DIR)/dgemm_recursively_blocked : override CFLAGS += -DDGEMM_RECURSIVELY_BLOCKED $(PAPI_INC) $(PAPI_LIB)

$(BIN_DIR)/two_scale_naive : override CFLAGS += -DTWO_SCALE_NAIVE
$(BIN_DIR)/two_scale_fused : override CFLAGS += -DTWO_SCALE_FUSED