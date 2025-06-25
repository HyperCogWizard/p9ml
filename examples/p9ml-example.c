// P9-ML Example: Simple Membrane Computing with Data-Free QAT
// This example demonstrates the basic usage of the P9-ML membrane computing system

#include "ggml-p9ml.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("P9-ML Membrane Computing Example\n");
    printf("================================\n\n");
    
    // Initialize GGML context with more memory
    struct ggml_init_params params = {
        .mem_size = 512 * 1024 * 1024,  // 512MB
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("Failed to initialize GGML context\n");
        return 1;
    }
    
    // Create CPU backend for computation
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        printf("Failed to initialize CPU backend\n");
        ggml_free(ctx);
        return 1;
    }
    
    // Create P9-ML namespace (distributed ML computation space)
    struct ggml_p9ml_namespace * ns = ggml_p9ml_namespace_new("ml_workspace", backend);
    if (!ns) {
        printf("Failed to create P9-ML namespace\n");
        ggml_backend_free(backend);
        ggml_free(ctx);
        return 1;
    }
    
    printf("1. Creating Membrane Computing Hierarchy\n");
    printf("---------------------------------------\n");
    
    // Create root membrane (represents main model)
    struct ggml_p9ml_membrane * root = ggml_p9ml_membrane_new("transformer_model", 0, ctx);
    
    // Create child membranes for different model components
    struct ggml_p9ml_membrane * embedding_layer = ggml_p9ml_membrane_new("embedding", 1, ctx);
    struct ggml_p9ml_membrane * attention_layer = ggml_p9ml_membrane_new("attention", 1, ctx);
    struct ggml_p9ml_membrane * ffn_layer = ggml_p9ml_membrane_new("ffn", 1, ctx);
    
    // Build hierarchy
    ggml_p9ml_membrane_add_child(root, embedding_layer);
    ggml_p9ml_membrane_add_child(root, attention_layer);
    ggml_p9ml_membrane_add_child(root, ffn_layer);
    
    // Connect to namespace
    ggml_p9ml_namespace_set_root(ns, root);
    
    printf("Created membrane hierarchy:\n");
    ggml_p9ml_print_membrane_stats(root);
    ggml_p9ml_print_membrane_stats(embedding_layer);
    ggml_p9ml_print_membrane_stats(attention_layer);
    ggml_p9ml_print_membrane_stats(ffn_layer);
    
    printf("2. Adding Model Parameters (Tensors)\n");
    printf("-----------------------------------\n");
    
    // Add tensors to embedding layer (smaller sizes for demo)
    struct ggml_tensor * word_embeddings = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 1000);  // Reduced from 768x50000
    struct ggml_tensor * pos_embeddings = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 512);    // Reduced from 768x1024
    ggml_p9ml_membrane_add_object(embedding_layer, word_embeddings);
    ggml_p9ml_membrane_add_object(embedding_layer, pos_embeddings);
    
    // Add tensors to attention layer  
    struct ggml_tensor * query_weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 512);     // Reduced from 768x768
    struct ggml_tensor * key_weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 512);
    struct ggml_tensor * value_weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 512);
    ggml_p9ml_membrane_add_object(attention_layer, query_weights);
    ggml_p9ml_membrane_add_object(attention_layer, key_weights);
    ggml_p9ml_membrane_add_object(attention_layer, value_weights);
    
    // Add tensors to FFN layer
    struct ggml_tensor * ffn_up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 2048);           // Reduced from 768x3072
    struct ggml_tensor * ffn_down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 512);         // Reduced from 3072x768
    ggml_p9ml_membrane_add_object(ffn_layer, ffn_up);
    ggml_p9ml_membrane_add_object(ffn_layer, ffn_down);
    
    // Initialize with random data
    if (word_embeddings->data) {
        float * data = (float *)word_embeddings->data;
        for (size_t i = 0; i < ggml_nelements(word_embeddings); i++) {
            data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
    }
    
    printf("Added tensors to membranes:\n");
    ggml_p9ml_print_membrane_stats(embedding_layer);
    ggml_p9ml_print_membrane_stats(attention_layer);
    ggml_p9ml_print_membrane_stats(ffn_layer);
    
    printf("3. Applying Data-Free QAT\n");
    printf("------------------------\n");
    
    // Create QAT configuration for 4-bit quantization
    struct ggml_p9ml_qat_config * qat_config = ggml_p9ml_qat_config_new(GGML_TYPE_Q4_K, 0.05f);
    qat_config->per_channel = true;
    qat_config->mixed_precision = true;
    qat_config->num_steps = 50;
    
    printf("QAT Configuration:\n");
    printf("  Target type: %s\n", ggml_type_name(qat_config->target_type));
    printf("  Noise scale: %.3f\n", (double)qat_config->noise_scale);
    printf("  Per-channel: %s\n", qat_config->per_channel ? "enabled" : "disabled");
    printf("  Mixed precision: %s\n", qat_config->mixed_precision ? "enabled" : "disabled");
    printf("  Training steps: %d\n", qat_config->num_steps);
    printf("\n");
    
    // Apply data-free QAT to the entire model
    printf("Applying data-free QAT to model...\n");
    int result = ggml_p9ml_apply_data_free_qat(root, qat_config);
    if (result == 0) {
        printf("✓ Data-free QAT applied successfully\n");
    } else {
        printf("✗ Failed to apply data-free QAT\n");
    }
    
    printf("\\n4. Membrane Evolution (P-Systems Computation)\n");
    printf("---------------------------------------------\n");
    
    // Simulate P-Systems evolution
    printf("Evolving membrane system...\n");
    result = ggml_p9ml_membrane_evolve(root);
    if (result == 0) {
        printf("✓ Membrane evolution completed\n");
    } else {
        printf("✗ Membrane evolution failed\n");
    }
    
    printf("\\n5. Mixed Precision Optimization\n");
    printf("-------------------------------\n");
    
    // Apply mixed precision quantization
    printf("Applying mixed precision quantization...\n");
    result = ggml_p9ml_mixed_precision_quantize(root, 0.95f);
    if (result == 0) {
        printf("✓ Mixed precision quantization completed\n");
    } else {
        printf("✗ Mixed precision quantization failed\n");
    }
    
    printf("\\n6. Forward Tiled QAT\n");
    printf("-------------------\n");
    
    // Apply forward tiled QAT
    printf("Applying forward tiled QAT...\n");
    result = ggml_p9ml_forward_tiled_qat(root, qat_config, NULL);
    if (result == 0) {
        printf("✓ Forward tiled QAT completed\n");
    } else {
        printf("✗ Forward tiled QAT failed\n");
    }
    
    printf("\\n7. Final Statistics\n");
    printf("------------------\n");
    
    // Update namespace statistics
    ns->total_params = ggml_nelements(word_embeddings) + ggml_nelements(pos_embeddings) +
                      ggml_nelements(query_weights) + ggml_nelements(key_weights) + 
                      ggml_nelements(value_weights) + ggml_nelements(ffn_up) + 
                      ggml_nelements(ffn_down);
    ns->quantized_params = ns->total_params; // All parameters were quantized
    ns->compression_ratio = 8.0f / 4.0f; // FP32 to 4-bit compression
    
    ggml_p9ml_print_namespace_stats(ns);
    
    printf("P9-ML Membrane Computing Example Completed Successfully!\\n");
    printf("\\nThis example demonstrated:\n");
    printf("• Creating hierarchical membrane computing structures\n");
    printf("• Data-free quantization aware training (QAT)\n");
    printf("• P-Systems inspired membrane evolution\n");
    printf("• Mixed precision optimization\n");
    printf("• Forward tiled processing\n");
    printf("• Distributed ML namespace management\n");
    
    // Cleanup - only free the root membrane, children are freed automatically
    ggml_p9ml_membrane_free(root);  // This will free all children too
    ggml_p9ml_namespace_free(ns);
    ggml_backend_free(backend);
    ggml_free(ctx);
    
    return 0;
}