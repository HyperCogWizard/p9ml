#pragma once

//
// P9-ML Systems: P-Systems to P9-ML-Systems
// Membrane Computing Framework for Distributed ML Namespaces
//
// This header implements a P-Systems inspired membrane computing framework
// for machine learning with data-free QAT (Quantization Aware Training) capabilities.
//

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// P9-ML Membrane System Types
typedef struct ggml_p9ml_membrane ggml_p9ml_membrane;
typedef struct ggml_p9ml_namespace ggml_p9ml_namespace;
typedef struct ggml_p9ml_qat_config ggml_p9ml_qat_config;

// Membrane Computing Abstraction
// Represents a computational membrane with rules and objects
struct ggml_p9ml_membrane {
    char name[64];                          // membrane identifier
    int level;                              // membrane hierarchy level
    struct ggml_context * ctx;              // GGML context for this membrane
    struct ggml_p9ml_namespace * ns;        // associated namespace
    struct ggml_p9ml_membrane * parent;     // parent membrane (NULL for root)
    struct ggml_p9ml_membrane ** children;  // child membranes
    int num_children;                       // number of child membranes
    int max_children;                       // maximum children capacity
    
    // Membrane objects (tensors)
    struct ggml_tensor ** objects;          // tensors in this membrane
    int num_objects;                        // number of objects
    int max_objects;                        // maximum objects capacity
    
    // Evolution rules (transformations)
    void ** rules;                          // function pointers for rules
    int num_rules;                          // number of rules
    int max_rules;                          // maximum rules capacity
    
    // Quantization context
    struct ggml_p9ml_qat_config * qat_config;
};

// Distributed Namespace Management
// Manages distributed ML computations across membranes
struct ggml_p9ml_namespace {
    char name[64];                          // namespace identifier
    struct ggml_p9ml_membrane * root;       // root membrane
    struct ggml_backend * backend;          // computation backend
    
    // Global namespace properties
    float noise_scale;                      // for data-free QAT
    int target_bits;                        // target quantization bits
    bool mixed_precision;                   // enable mixed precision
    
    // Performance metrics
    size_t total_params;                    // total parameters
    size_t quantized_params;                // quantized parameters
    float compression_ratio;                // achieved compression
};

// Data-Free QAT Configuration
// Configuration for data-free quantization aware training
struct ggml_p9ml_qat_config {
    enum ggml_type target_type;             // target quantization type
    float noise_scale;                      // noise injection scale
    bool per_channel;                       // per-channel quantization
    bool mixed_precision;                   // mixed precision mode
    
    // Advanced QAT parameters
    float temperature;                      // for soft quantization
    int num_steps;                          // training steps
    float learning_rate;                    // learning rate for QAT
    
    // Forward tiled QAT
    int tile_size;                          // tile size for forward QAT
    bool use_reference;                     // use FP reference
};

// P9-ML System API

// Membrane creation and management
GGML_API struct ggml_p9ml_membrane * ggml_p9ml_membrane_new(
    const char * name,
    int level,
    struct ggml_context * ctx);

GGML_API void ggml_p9ml_membrane_free(struct ggml_p9ml_membrane * membrane);

GGML_API int ggml_p9ml_membrane_add_child(
    struct ggml_p9ml_membrane * parent,
    struct ggml_p9ml_membrane * child);

GGML_API int ggml_p9ml_membrane_add_object(
    struct ggml_p9ml_membrane * membrane,
    struct ggml_tensor * tensor);

// Namespace management
GGML_API struct ggml_p9ml_namespace * ggml_p9ml_namespace_new(
    const char * name,
    struct ggml_backend * backend);

GGML_API void ggml_p9ml_namespace_free(struct ggml_p9ml_namespace * ns);

GGML_API int ggml_p9ml_namespace_set_root(
    struct ggml_p9ml_namespace * ns,
    struct ggml_p9ml_membrane * root);

// Data-Free QAT Functions
GGML_API struct ggml_p9ml_qat_config * ggml_p9ml_qat_config_new(
    enum ggml_type target_type,
    float noise_scale);

GGML_API void ggml_p9ml_qat_config_free(struct ggml_p9ml_qat_config * config);

GGML_API int ggml_p9ml_apply_data_free_qat(
    struct ggml_p9ml_membrane * membrane,
    struct ggml_p9ml_qat_config * config);

// Generate synthetic data for data-free training
GGML_API struct ggml_tensor * ggml_p9ml_generate_synthetic_data(
    struct ggml_context * ctx,
    const int64_t * shape,
    int n_dims,
    float noise_scale);

// Forward tiled QAT
GGML_API int ggml_p9ml_forward_tiled_qat(
    struct ggml_p9ml_membrane * membrane,
    struct ggml_p9ml_qat_config * config,
    struct ggml_tensor * reference);

// Mixed-precision quantization
GGML_API int ggml_p9ml_mixed_precision_quantize(
    struct ggml_p9ml_membrane * membrane,
    float quality_threshold);

// Membrane evolution (P-Systems computation)
GGML_API int ggml_p9ml_membrane_evolve(struct ggml_p9ml_membrane * membrane);

// Distributed computation across namespace
GGML_API int ggml_p9ml_namespace_compute(
    struct ggml_p9ml_namespace * ns,
    struct ggml_cgraph * graph);

// Utility functions
GGML_API void ggml_p9ml_print_membrane_stats(struct ggml_p9ml_membrane * membrane);
GGML_API void ggml_p9ml_print_namespace_stats(struct ggml_p9ml_namespace * ns);

#ifdef __cplusplus
}
#endif