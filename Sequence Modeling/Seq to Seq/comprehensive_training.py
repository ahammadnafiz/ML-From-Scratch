import numpy as np
import matplotlib.pyplot as plt
from encoder_decoder import GRUEncoderDecoder

def comprehensive_training_demo():
    """
    Comprehensive demonstration of all GRU Encoder-Decoder functionality.
    """
    print("=" * 80)
    print("COMPREHENSIVE GRU ENCODER-DECODER TRAINING DEMONSTRATION")
    print("=" * 80)
    
    # Initialize model with custom parameters
    print("\n1. INITIALIZING MODEL")
    print("-" * 40)
    
    model = GRUEncoderDecoder(
        input_dim=8,        # Reduced for faster training
        hidden_dim=16,      # Reduced for faster training  
        output_dim=8,       # Same as input for autoencoder-like task
        vocab_size=10,      # Small vocabulary
        beta1=0.9,          # Adam momentum parameter
        beta2=0.999,        # Adam RMSprop parameter
        epsilon=1e-8        # Adam numerical stability
    )
    
    print(f"✓ Model initialized successfully!")
    print(f"  - Input dimension: {model.input_dim}")
    print(f"  - Hidden dimension: {model.hidden_dim}")
    print(f"  - Output dimension: {model.output_dim}")
    print(f"  - Vocabulary size: {model.vocab_size}")
    print(f"  - Adam parameters: β₁={model.beta1}, β₂={model.beta2}, ε={model.epsilon}")
    
    # 2. Generate structured training data (USING UNUSED METHOD)
    print("\n2. GENERATING STRUCTURED TRAINING DATA")
    print("-" * 40)
    
    X_train, Y_train, targets_train = model.generate_structured_data(
        seq_len=12, 
        target_len=8, 
        batch_size=1
    )
    
    print(f"✓ Structured training data generated!")
    print(f"  - Input sequence shape: {X_train.shape}")
    print(f"  - Decoder input shape: {Y_train.shape}")
    print(f"  - Target shape: {targets_train.shape}")
    print(f"  - Input pattern preview: {np.argmax(X_train, axis=0)}")
    print(f"  - Target pattern preview: {np.argmax(targets_train, axis=0)}")
    
    # Generate validation data
    X_val, Y_val, targets_val = model.generate_structured_data(
        seq_len=12, 
        target_len=8, 
        batch_size=1
    )
    
    # 3. Initial evaluation (USING UNUSED METHOD)
    print("\n3. INITIAL MODEL EVALUATION")
    print("-" * 40)
    
    initial_metrics = model.evaluate_model(X_train, Y_train, targets_train)
    
    print(f"✓ Initial evaluation completed!")
    print(f"  - Initial loss: {initial_metrics['loss']:.4f}")
    print(f"  - Initial accuracy: {initial_metrics['accuracy']:.4f}")
    print(f"  - Initial perplexity: {initial_metrics['perplexity']:.4f}")
    
    # 4. Training with learning rate scheduling (USING UNUSED METHOD)
    print("\n4. TRAINING WITH ADAPTIVE LEARNING RATE")
    print("-" * 40)
    
    num_epochs = 1000
    initial_lr = 0.01
    
    # Storage for metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    learning_rates = []
    
    print("Starting training with learning rate scheduling...")
    print("Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Learning Rate")
    print("-" * 70)
    
    for epoch in range(num_epochs):
        # Get scheduled learning rate (USING UNUSED METHOD)
        current_lr = model.lr_schedule(epoch, initial_lr)
        learning_rates.append(current_lr)
        
        # Training step
        train_loss = model.train_step(X_train, Y_train, targets_train, current_lr)
        
        # Evaluate on training data (USING UNUSED METHOD)
        train_metrics = model.evaluate_model(X_train, Y_train, targets_train)
        
        # Evaluate on validation data (USING UNUSED METHOD)  
        val_metrics = model.evaluate_model(X_val, Y_val, targets_val)
        
        # Store metrics
        train_losses.append(train_metrics['loss'])
        val_losses.append(val_metrics['loss'])
        train_accuracies.append(train_metrics['accuracy'])
        val_accuracies.append(val_metrics['accuracy'])
        
        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch < 5 or epoch >= num_epochs - 5:
            print(f"{epoch+1:5d} | {train_metrics['loss']:10.4f} | {val_metrics['loss']:8.4f} | "
                  f"{train_metrics['accuracy']:9.4f} | {val_metrics['accuracy']:7.4f} | {current_lr:13.6f}")
    
    # 5. Final evaluation and comparison
    print("\n5. FINAL EVALUATION AND COMPARISON")
    print("-" * 40)
    
    final_metrics = model.evaluate_model(X_train, Y_train, targets_train)
    
    print(f"✓ Training completed successfully!")
    print(f"\nPerformance Improvement:")
    print(f"  - Loss: {initial_metrics['loss']:.4f} → {final_metrics['loss']:.4f} "
          f"({((initial_metrics['loss'] - final_metrics['loss']) / initial_metrics['loss'] * 100):+.2f}%)")
    print(f"  - Accuracy: {initial_metrics['accuracy']:.4f} → {final_metrics['accuracy']:.4f} "
          f"({((final_metrics['accuracy'] - initial_metrics['accuracy']) * 100):+.2f}%)")
    print(f"  - Perplexity: {initial_metrics['perplexity']:.4f} → {final_metrics['perplexity']:.4f}")
    
    # 6. Test prediction capability
    print("\n6. TESTING PREDICTION CAPABILITY")
    print("-" * 40)
    
    # Generate test sequence
    test_input = np.random.rand(model.input_dim, 10)
    generated_sequence = model.predict(test_input, max_length=8)
    
    print(f"✓ Prediction test completed!")
    print(f"  - Test input shape: {test_input.shape}")
    print(f"  - Generated sequence: {generated_sequence}")
    print(f"  - Generated length: {len(generated_sequence)}")
    
    # 7. Visualize training progress
    print("\n7. VISUALIZING TRAINING PROGRESS")
    print("-" * 40)
    
    try:
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Loss curves
        ax1.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='blue')
        ax1.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy curves
        ax2.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy', color='blue')
        ax2.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Learning rate schedule
        ax3.plot(range(1, num_epochs + 1), learning_rates, color='green', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Perplexity
        train_perplexities = [np.exp(loss) for loss in train_losses]
        val_perplexities = [np.exp(loss) for loss in val_losses]
        ax4.plot(range(1, num_epochs + 1), train_perplexities, label='Training Perplexity', color='blue')
        ax4.plot(range(1, num_epochs + 1), val_perplexities, label='Validation Perplexity', color='red')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Perplexity')
        ax4.set_title('Training and Validation Perplexity')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/media/nafiz/NewVolume/ML-From-Scratch/Sequence Modeling/Seq to Seq/training_progress.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Training visualization saved as 'training_progress.png'")
        
    except Exception as e:
        print(f"⚠ Visualization failed (matplotlib not available): {e}")
    
    # 8. Demonstrate advanced features
    print("\n8. ADVANCED FEATURES DEMONSTRATION")
    print("-" * 40)
    
    # Test gradient checking
    context, enc_caches = model.encoder_forward(X_train)
    predictions, dec_caches, init_cache = model.decoder_forward(Y_train, context)
    gradients = model.backward(predictions, targets_train, enc_caches, dec_caches, init_cache)
    
    gradient_check_passed = model.check_gradients(gradients)
    print(f"✓ Gradient check: {'PASSED' if gradient_check_passed else 'FAILED'}")
    
    # Test gradient clipping
    clipped_gradients = model.clip_gradients(gradients.copy(), max_norm=1.0)
    print(f"✓ Gradient clipping applied")
    
    # Show Adam optimizer state
    print(f"✓ Adam optimizer state:")
    print(f"  - Current time step: {model.t}")
    print(f"  - Bias correction factors: {1 - model.beta1**model.t:.6f}, {1 - model.beta2**model.t:.6f}")
    
    # 9. Generate multiple structured datasets
    print("\n9. MULTIPLE DATASET GENERATION")
    print("-" * 40)
    
    datasets = []
    for i in range(3):
        X, Y, targets = model.generate_structured_data(seq_len=8+i*2, target_len=6+i, batch_size=1)
        datasets.append((X, Y, targets))
        print(f"✓ Dataset {i+1}: Input({X.shape}), Target({targets.shape})")
    
    # 10. Summary and cleanup
    print("\n10. SUMMARY")
    print("-" * 40)
    
    print(f"✓ Comprehensive training demonstration completed successfully!")
    print(f"\nMethods utilized:")
    print(f"  ✓ lr_schedule: Learning rate scheduling with exponential decay")
    print(f"  ✓ generate_structured_data: Structured training data generation")
    print(f"  ✓ evaluate_model: Comprehensive model evaluation with metrics")
    print(f"  ✓ check_gradients: Gradient sanity checking")
    print(f"  ✓ clip_gradients: Gradient clipping for stability")
    print(f"  ✓ adam_update: Adam optimizer with bias correction")
    print(f"  ✓ predict: Sequence generation with greedy decoding")
    
    print(f"\nFinal Statistics:")
    print(f"  - Total training epochs: {num_epochs}")
    print(f"  - Final training loss: {final_metrics['loss']:.4f}")
    print(f"  - Final training accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  - Final learning rate: {learning_rates[-1]:.6f}")
    print(f"  - Adam time steps: {model.t}")
    
    return model, (train_losses, val_losses, train_accuracies, val_accuracies, learning_rates)

def extended_functionality_demo():
    """
    Extended demonstration of additional functionality.
    """
    print("\n" + "=" * 80)
    print("EXTENDED FUNCTIONALITY DEMONSTRATION")
    print("=" * 80)
    
    # Create a different model configuration
    model = GRUEncoderDecoder(
        input_dim=6,
        hidden_dim=12,
        output_dim=6,
        vocab_size=15,
        beta1=0.95,  # Different Adam parameters
        beta2=0.999,
        epsilon=1e-7
    )
    
    # 1. Test multiple sequence lengths
    print("\n1. TESTING MULTIPLE SEQUENCE LENGTHS")
    print("-" * 40)
    
    for seq_len in [5, 10, 15]:
        X, Y, targets = model.generate_structured_data(seq_len=seq_len, target_len=seq_len//2)
        metrics = model.evaluate_model(X, Y, targets)
        print(f"Seq length {seq_len:2d}: Loss={metrics['loss']:6.4f}, Acc={metrics['accuracy']:6.4f}")
    
    # 2. Test different learning rate schedules
    print("\n2. TESTING LEARNING RATE SCHEDULES")
    print("-" * 40)
    
    for epoch in [0, 10, 20, 50, 100]:
        lr = model.lr_schedule(epoch, initial_lr=0.01)
        print(f"Epoch {epoch:3d}: Learning rate = {lr:.6f}")
    
    # 3. Test prediction with different max lengths
    print("\n3. TESTING PREDICTION LENGTHS")
    print("-" * 40)
    
    test_input = np.random.rand(6, 8)
    for max_len in [5, 10, 15]:
        generated = model.predict(test_input, max_length=max_len)
        print(f"Max length {max_len:2d}: Generated {len(generated)} tokens: {generated}")
    
    print(f"\n✓ Extended functionality demonstration completed!")

if __name__ == "__main__":
    """
    Main execution: Run comprehensive training demonstration.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        # Run comprehensive demonstration
        model, metrics = comprehensive_training_demo()
        
        # Run extended functionality demonstration
        extended_functionality_demo()
        
        print("\n" + "=" * 80)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("• lr_schedule() - Used for adaptive learning rate scheduling")
        print("• generate_structured_data() - Used for creating structured training data")
        print("• evaluate_model() - Used for comprehensive model evaluation")
        print("\nAdditionally demonstrated:")
        print("• Comprehensive training loop with validation")
        print("• Training progress visualization")
        print("• Gradient checking and clipping")
        print("• Adam optimizer monitoring")
        print("• Multiple dataset generation")
        print("• Extended functionality testing")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
