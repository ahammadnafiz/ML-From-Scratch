#!/usr/bin/env python3
"""
Translation Training Script for GRU Encoder-Decoder Model
=========================================================

This script trains and tests the GRU Encoder-Decoder model on a popular translation dataset.
We'll use the Multi30K German-English translation dataset, which is widely used for
neural machine translation research.

Dataset: Multi30K (German -> English)
- Training: 29,000 sentence pairs
- Validation: 1,014 sentence pairs  
- Test: 1,000 sentence pairs
- Domain: Image descriptions (Flickr30K entities)
- Source: http://www.statmt.org/wmt16/multimodal-task.html

"""

import os
import re
import requests
import zipfile
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional
import time

# Import our GRU Encoder-Decoder model
from encoder_decoder import GRUEncoderDecoder


class TranslationDataset:
    """
    Dataset handler for translation tasks with vocabulary management and preprocessing.
    """
    
    def __init__(self, max_vocab_size: int = 10000, max_seq_len: int = 20):
        """
        Initialize the translation dataset handler.
        
        Args:
            max_vocab_size: Maximum vocabulary size for both source and target
            max_seq_len: Maximum sequence length (longer sequences will be truncated)
        """
        self.max_vocab_size = max_vocab_size
        self.max_seq_len = max_seq_len
        
        # Vocabulary mappings
        self.src_vocab = {}  # word -> id
        self.tgt_vocab = {}  # word -> id
        self.src_vocab_inv = {}  # id -> word
        self.tgt_vocab_inv = {}  # id -> word
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.SOS_TOKEN = '<SOS>'  # Start of sequence
        self.EOS_TOKEN = '<EOS>'  # End of sequence
        
        # Special token IDs (must be first 4 IDs)
        self.PAD_ID = 0
        self.UNK_ID = 1
        self.SOS_ID = 2
        self.EOS_ID = 3
        
    def download_multi30k_dataset(self, data_dir: str = 'data'):
        """
        Download the Multi30K German-English dataset.
        """
        os.makedirs(data_dir, exist_ok=True)
        
        print("📥 Downloading Multi30K German-English dataset...")
        
        # URLs for Multi30K dataset files
        base_url = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/"
        
        files = [
            ("train.de.gz", "train.de"),
            ("train.en.gz", "train.en"), 
            ("val.de.gz", "val.de"),
            ("val.en.gz", "val.en"),
            ("test_2016_flickr.de.gz", "test.de"),
            ("test_2016_flickr.en.gz", "test.en")
        ]
        
        for gz_file, txt_file in files:
            url = base_url + gz_file
            gz_path = os.path.join(data_dir, gz_file)
            txt_path = os.path.join(data_dir, txt_file)
            
            if not os.path.exists(txt_path):
                print(f"  Downloading {gz_file}...")
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    
                    with open(gz_path, 'wb') as f:
                        f.write(response.content)
                    
                    # Extract gz file
                    import gzip
                    with gzip.open(gz_path, 'rt', encoding='utf-8') as f_in:
                        with open(txt_path, 'w', encoding='utf-8') as f_out:
                            f_out.write(f_in.read())
                    
                    # Remove gz file
                    os.remove(gz_path)
                    print(f"  ✅ {txt_file} ready")
                    
                except Exception as e:
                    print(f"  ❌ Failed to download {gz_file}: {e}")
                    # Create a small sample dataset for demonstration
                    print(f"  📝 Creating sample dataset for {txt_file}...")
                    self._create_sample_data(txt_path, gz_file)
            else:
                print(f"  ✅ {txt_file} already exists")
    
    def _create_sample_data(self, file_path: str, original_name: str):
        """Create sample translation data for demonstration if download fails."""
        if 'de' in original_name:  # German samples
            sample_data = [
                "Ein Mann sitzt auf einer Bank.",
                "Eine Frau läuft im Park.",
                "Das Kind spielt mit einem Ball.",
                "Der Hund rennt schnell.",
                "Die Katze schläft auf dem Sofa.",
                "Ein Auto fährt auf der Straße.",
                "Der Vogel fliegt hoch am Himmel.",
                "Die Blumen blühen im Garten.",
                "Ein Buch liegt auf dem Tisch.",
                "Der Mann trägt eine Brille.",
                "Die Frau hat lange Haare.",
                "Das Haus ist sehr groß.",
                "Der Bus kommt jeden Morgen.",
                "Die Kinder spielen zusammen.",
                "Ein Fahrrad steht vor der Tür."
            ]
        else:  # English samples
            sample_data = [
                "A man sits on a bench.",
                "A woman runs in the park.",
                "The child plays with a ball.",
                "The dog runs quickly.",
                "The cat sleeps on the sofa.",
                "A car drives on the street.",
                "The bird flies high in the sky.",
                "The flowers bloom in the garden.",
                "A book lies on the table.",
                "The man wears glasses.",
                "The woman has long hair.",
                "The house is very big.",
                "The bus comes every morning.",
                "The children play together.",
                "A bicycle stands in front of the door."
            ]
        
        # Extend to have more samples
        extended_data = sample_data * 20  # Repeat 20 times for ~300 samples
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in extended_data:
                f.write(line + '\n')
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by lowercasing, removing special characters, etc.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Add spaces around punctuation
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        
        # Remove extra spaces again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def build_vocabulary(self, src_sentences: List[str], tgt_sentences: List[str]):
        """
        Build vocabularies for source and target languages.
        """
        print("🔤 Building vocabularies...")
        
        # Count word frequencies
        src_counter = Counter()
        tgt_counter = Counter()
        
        for sentence in src_sentences:
            words = self.preprocess_text(sentence).split()
            src_counter.update(words)
        
        for sentence in tgt_sentences:
            words = self.preprocess_text(sentence).split()
            tgt_counter.update(words)
        
        # Build source vocabulary
        self.src_vocab = {
            self.PAD_TOKEN: self.PAD_ID,
            self.UNK_TOKEN: self.UNK_ID,
            self.SOS_TOKEN: self.SOS_ID,
            self.EOS_TOKEN: self.EOS_ID
        }
        
        # Add most frequent words
        for word, count in src_counter.most_common(self.max_vocab_size - 4):
            if word not in self.src_vocab:
                self.src_vocab[word] = len(self.src_vocab)
        
        # Build target vocabulary
        self.tgt_vocab = {
            self.PAD_TOKEN: self.PAD_ID,
            self.UNK_TOKEN: self.UNK_ID,
            self.SOS_TOKEN: self.SOS_ID,
            self.EOS_TOKEN: self.EOS_ID
        }
        
        for word, count in tgt_counter.most_common(self.max_vocab_size - 4):
            if word not in self.tgt_vocab:
                self.tgt_vocab[word] = len(self.tgt_vocab)
        
        # Build inverse vocabularies
        self.src_vocab_inv = {v: k for k, v in self.src_vocab.items()}
        self.tgt_vocab_inv = {v: k for k, v in self.tgt_vocab.items()}
        
        print(f"  📊 Source vocabulary size: {len(self.src_vocab)}")
        print(f"  📊 Target vocabulary size: {len(self.tgt_vocab)}")
        
    def sentence_to_ids(self, sentence: str, vocab: Dict[str, int], add_eos: bool = False) -> List[int]:
        """
        Convert a sentence to a list of token IDs.
        """
        words = self.preprocess_text(sentence).split()[:self.max_seq_len-2]  # Leave room for SOS/EOS
        ids = [vocab.get(word, self.UNK_ID) for word in words]
        
        if add_eos:
            ids.append(self.EOS_ID)
            
        return ids
    
    def ids_to_sentence(self, ids: List[int], vocab_inv: Dict[int, str]) -> str:
        """
        Convert a list of token IDs back to a sentence.
        """
        words = []
        for id in ids:
            word = vocab_inv.get(id, self.UNK_TOKEN)
            if word in [self.PAD_TOKEN, self.EOS_TOKEN]:
                break
            if word != self.SOS_TOKEN:
                words.append(word)
        return ' '.join(words)
    
    def load_data(self, data_dir: str = 'data') -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Load and preprocess the translation dataset.
        
        Returns:
            train_pairs, val_pairs, test_pairs: Lists of (source, target) sentence pairs
        """
        print("📂 Loading translation dataset...")
        
        # Download data if not available
        self.download_multi30k_dataset(data_dir)
        
        def load_sentence_pairs(src_file: str, tgt_file: str) -> List[Tuple[str, str]]:
            pairs = []
            try:
                with open(src_file, 'r', encoding='utf-8') as f_src, \
                     open(tgt_file, 'r', encoding='utf-8') as f_tgt:
                    
                    for src_line, tgt_line in zip(f_src, f_tgt):
                        src_line = src_line.strip()
                        tgt_line = tgt_line.strip()
                        
                        if src_line and tgt_line:  # Skip empty lines
                            pairs.append((src_line, tgt_line))
                            
            except FileNotFoundError as e:
                print(f"⚠️  File not found: {e}")
                return []
            
            return pairs
        
        # Load training, validation, and test data
        train_pairs = load_sentence_pairs(
            os.path.join(data_dir, 'train.de'),
            os.path.join(data_dir, 'train.en')
        )
        
        val_pairs = load_sentence_pairs(
            os.path.join(data_dir, 'val.de'),
            os.path.join(data_dir, 'val.en')
        )
        
        test_pairs = load_sentence_pairs(
            os.path.join(data_dir, 'test.de'),
            os.path.join(data_dir, 'test.en')
        )
        
        print(f"  📊 Training pairs: {len(train_pairs)}")
        print(f"  📊 Validation pairs: {len(val_pairs)}")
        print(f"  📊 Test pairs: {len(test_pairs)}")
        
        # Build vocabularies using training data
        if train_pairs:
            src_sentences = [pair[0] for pair in train_pairs]
            tgt_sentences = [pair[1] for pair in train_pairs]
            self.build_vocabulary(src_sentences, tgt_sentences)
        
        return train_pairs, val_pairs, test_pairs


class TranslationTrainer:
    """
    Trainer class for the GRU Encoder-Decoder translation model.
    """
    
    def __init__(self, dataset: TranslationDataset, model: GRUEncoderDecoder):
        """
        Initialize the trainer.
        
        Args:
            dataset: TranslationDataset instance
            model: GRUEncoderDecoder model instance
        """
        self.dataset = dataset
        self.model = model
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_bleu': [],
            'val_bleu': [],
            'epochs': []
        }
    
    def prepare_batch(self, sentence_pairs: List[Tuple[str, str]], batch_size: int = 1) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Prepare batches of data for training.
        
        Returns:
            List of (encoder_input, decoder_input, targets) tuples
        """
        batches = []
        
        for i in range(0, len(sentence_pairs), batch_size):
            batch_pairs = sentence_pairs[i:i + batch_size]
            
            if len(batch_pairs) == 0:
                continue
            
            # For simplicity, we'll use batch_size=1 for now
            # In practice, you'd want to implement proper batching with padding
            src_sentence, tgt_sentence = batch_pairs[0]
            
            # Convert to token IDs
            src_ids = self.dataset.sentence_to_ids(src_sentence, self.dataset.src_vocab)
            tgt_ids = self.dataset.sentence_to_ids(tgt_sentence, self.dataset.tgt_vocab, add_eos=True)
            
            # Create one-hot encodings
            encoder_input = np.zeros((len(self.dataset.src_vocab), len(src_ids)))
            for t, token_id in enumerate(src_ids):
                if t < encoder_input.shape[1]:
                    encoder_input[token_id, t] = 1.0
            
            # Decoder input (starts with SOS token)
            decoder_input_ids = [self.dataset.SOS_ID] + tgt_ids[:-1]  # Shift right
            decoder_input = np.zeros((len(self.dataset.tgt_vocab), len(decoder_input_ids)))
            for t, token_id in enumerate(decoder_input_ids):
                if t < decoder_input.shape[1]:
                    decoder_input[token_id, t] = 1.0
            
            # Targets (what we want to predict)
            targets = np.zeros((len(self.dataset.tgt_vocab), len(tgt_ids)))
            for t, token_id in enumerate(tgt_ids):
                if t < targets.shape[1]:
                    targets[token_id, t] = 1.0
            
            batches.append((encoder_input, decoder_input, targets))
        
        return batches
    
    def calculate_bleu_score(self, generated_ids: List[int], reference_ids: List[int]) -> float:
        """
        Simple BLEU score calculation (1-gram precision for simplicity).
        """
        if len(generated_ids) == 0:
            return 0.0
        
        # Remove special tokens
        gen_clean = [id for id in generated_ids if id not in [self.dataset.PAD_ID, self.dataset.SOS_ID, self.dataset.EOS_ID]]
        ref_clean = [id for id in reference_ids if id not in [self.dataset.PAD_ID, self.dataset.SOS_ID, self.dataset.EOS_ID]]
        
        if len(gen_clean) == 0 or len(ref_clean) == 0:
            return 0.0
        
        # Calculate 1-gram precision
        matches = sum(1 for token in gen_clean if token in ref_clean)
        precision = matches / len(gen_clean)
        
        # Add brevity penalty
        bp = min(1.0, len(gen_clean) / len(ref_clean)) if len(ref_clean) > 0 else 0.0
        
        return bp * precision
    
    def evaluate(self, data_pairs: List[Tuple[str, str]], max_samples: int = 100) -> Tuple[float, float]:
        """
        Evaluate the model on a dataset.
        
        Returns:
            average_loss, average_bleu_score
        """
        if not data_pairs:
            return float('inf'), 0.0
        
        eval_pairs = data_pairs[:max_samples]  # Limit for faster evaluation
        total_loss = 0.0
        total_bleu = 0.0
        valid_samples = 0
        
        for src_sentence, tgt_sentence in eval_pairs:
            try:
                # Prepare data
                batches = self.prepare_batch([(src_sentence, tgt_sentence)])
                if not batches:
                    continue
                
                encoder_input, decoder_input, targets = batches[0]
                
                # Forward pass for loss
                context, _ = self.model.encoder_forward(encoder_input)
                predictions, _, _ = self.model.decoder_forward(decoder_input, context)
                loss = self.model.compute_loss(predictions, targets)
                
                # Generate translation for BLEU score
                generated_ids = self.model.predict(encoder_input, max_length=targets.shape[1])
                reference_ids = [np.argmax(targets[:, t]) for t in range(targets.shape[1])]
                
                bleu_score = self.calculate_bleu_score(generated_ids, reference_ids)
                
                total_loss += loss
                total_bleu += bleu_score
                valid_samples += 1
                
            except Exception as e:
                print(f"⚠️  Evaluation error: {e}")
                continue
        
        if valid_samples == 0:
            return float('inf'), 0.0
        
        avg_loss = total_loss / valid_samples
        avg_bleu = total_bleu / valid_samples
        
        return avg_loss, avg_bleu
    
    def train(self, train_pairs: List[Tuple[str, str]], val_pairs: List[Tuple[str, str]], 
              epochs: int = 100, learning_rate: float = 0.001, 
              eval_every: int = 10, save_every: int = 50) -> Dict:
        """
        Train the translation model.
        
        Args:
            train_pairs: Training sentence pairs
            val_pairs: Validation sentence pairs
            epochs: Number of training epochs
            learning_rate: Initial learning rate
            eval_every: Evaluate every N epochs
            save_every: Save model every N epochs
            
        Returns:
            Dictionary with training statistics
        """
        print(f"🚂 Starting training for {epochs} epochs...")
        print(f"   📚 Training samples: {len(train_pairs)}")
        print(f"   📝 Validation samples: {len(val_pairs)}")
        
        # Limit training data for manageable training time
        max_train_samples = min(1000, len(train_pairs))
        train_subset = train_pairs[:max_train_samples]
        
        print(f"   🎯 Using {max_train_samples} training samples for faster training")
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_losses = []
            
            # Shuffle training data
            np.random.shuffle(train_subset)
            
            # Training loop
            for i, (src_sentence, tgt_sentence) in enumerate(train_subset):
                try:
                    # Prepare batch
                    batches = self.prepare_batch([(src_sentence, tgt_sentence)])
                    if not batches:
                        continue
                    
                    encoder_input, decoder_input, targets = batches[0]
                    
                    # Adaptive learning rate
                    current_lr = self.model.lr_schedule(epoch, learning_rate)
                    
                    # Training step
                    loss = self.model.train_step(encoder_input, decoder_input, targets, current_lr)
                    epoch_losses.append(loss)
                    
                    # Print progress
                    if (i + 1) % 100 == 0 or i == len(train_subset) - 1:
                        avg_loss = np.mean(epoch_losses[-100:]) if epoch_losses else 0
                        print(f"    Epoch {epoch+1}/{epochs}, Sample {i+1}/{len(train_subset)}, "
                              f"Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
                
                except Exception as e:
                    print(f"⚠️  Training error at epoch {epoch+1}, sample {i+1}: {e}")
                    continue
            
            # Calculate epoch statistics
            if epoch_losses:
                avg_train_loss = np.mean(epoch_losses)
                self.training_history['train_loss'].append(avg_train_loss)
            else:
                avg_train_loss = float('inf')
                self.training_history['train_loss'].append(avg_train_loss)
            
            # Evaluation
            if (epoch + 1) % eval_every == 0 or epoch == 0:
                print(f"    📊 Evaluating at epoch {epoch+1}...")
                
                # Evaluate on validation set
                val_loss, val_bleu = self.evaluate(val_pairs, max_samples=50)
                train_loss_eval, train_bleu = self.evaluate(train_subset[:50], max_samples=50)
                
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_bleu'].append(val_bleu)
                self.training_history['train_bleu'].append(train_bleu)
                self.training_history['epochs'].append(epoch + 1)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"    🎉 New best validation loss: {val_loss:.4f}")
                
                epoch_time = time.time() - epoch_start
                print(f"    📈 Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
                      f"Val Loss={val_loss:.4f}, Train BLEU={train_bleu:.4f}, "
                      f"Val BLEU={val_bleu:.4f}, Time={epoch_time:.2f}s")
            
            # Save model periodically
            if (epoch + 1) % save_every == 0:
                self.save_model(f"model_epoch_{epoch+1}.pkl")
        
        total_time = time.time() - start_time
        print(f"🏁 Training completed in {total_time:.2f} seconds!")
        
        return self.training_history
    
    def save_model(self, filename: str):
        """Save the trained model and dataset."""
        model_data = {
            'model_state': {
                'W_r_enc': self.model.W_r_enc, 'W_z_enc': self.model.W_z_enc, 'W_h_enc': self.model.W_h_enc,
                'U_r_enc': self.model.U_r_enc, 'U_z_enc': self.model.U_z_enc, 'U_h_enc': self.model.U_h_enc,
                'b_r_enc': self.model.b_r_enc, 'b_z_enc': self.model.b_z_enc, 'b_h_enc': self.model.b_h_enc,
                'W_r_dec': self.model.W_r_dec, 'W_z_dec': self.model.W_z_dec, 'W_h_dec': self.model.W_h_dec,
                'U_r_dec': self.model.U_r_dec, 'U_z_dec': self.model.U_z_dec, 'U_h_dec': self.model.U_h_dec,
                'b_r_dec': self.model.b_r_dec, 'b_z_dec': self.model.b_z_dec, 'b_h_dec': self.model.b_h_dec,
                'W_o': self.model.W_o, 'b_o': self.model.b_o, 'W_c': self.model.W_c, 'b_c': self.model.b_c,
            },
            'model_config': {
                'input_dim': self.model.input_dim,
                'hidden_dim': self.model.hidden_dim,
                'output_dim': self.model.output_dim,
                'vocab_size': self.model.vocab_size
            },
            'dataset_vocab': {
                'src_vocab': self.dataset.src_vocab,
                'tgt_vocab': self.dataset.tgt_vocab,
                'src_vocab_inv': self.dataset.src_vocab_inv,
                'tgt_vocab_inv': self.dataset.tgt_vocab_inv
            },
            'training_history': self.training_history
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"💾 Model saved to {filename}")
    
    def test_translation(self, test_pairs: List[Tuple[str, str]], num_examples: int = 10):
        """Test the model on some examples and show translations."""
        print(f"🧪 Testing translation on {num_examples} examples...")
        
        for i, (src_sentence, tgt_sentence) in enumerate(test_pairs[:num_examples]):
            try:
                # Prepare input
                batches = self.prepare_batch([(src_sentence, tgt_sentence)])
                if not batches:
                    continue
                
                encoder_input, _, _ = batches[0]
                
                # Generate translation
                generated_ids = self.model.predict(encoder_input, max_length=20)
                generated_sentence = self.dataset.ids_to_sentence(generated_ids, self.dataset.tgt_vocab_inv)
                
                print(f"\n--- Example {i+1} ---")
                print(f"🇩🇪 German:     {src_sentence}")
                print(f"🇬🇧 Reference:  {tgt_sentence}")
                print(f"🤖 Generated:   {generated_sentence}")
                
                # Calculate BLEU score
                tgt_ids = self.dataset.sentence_to_ids(tgt_sentence, self.dataset.tgt_vocab, add_eos=True)
                bleu = self.calculate_bleu_score(generated_ids, tgt_ids)
                print(f"📊 BLEU Score:  {bleu:.4f}")
                
            except Exception as e:
                print(f"⚠️  Translation error for example {i+1}: {e}")
                continue


def plot_training_history(history: Dict):
    """Plot training curves."""
    if not history['epochs']:
        print("No training history to plot.")
        return
    
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(history['epochs'], history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(history['epochs'], history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot BLEU scores
    ax2.plot(history['epochs'], history['train_bleu'], 'b-', label='Training BLEU', linewidth=2)
    ax2.plot(history['epochs'], history['val_bleu'], 'r-', label='Validation BLEU', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('BLEU Score')
    ax2.set_title('Training and Validation BLEU Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("📈 Training curves saved as 'training_curves.png'")


def main():
    """Main training and testing pipeline."""
    print("=" * 80)
    print("🌍 GRU ENCODER-DECODER TRANSLATION TRAINING")
    print("   Dataset: Multi30K German → English")
    print("   Model: GRU Encoder-Decoder (From Scratch)")
    print("=" * 80)
    
    # Configuration
    config = {
        'max_vocab_size': 5000,
        'max_seq_len': 15,
        'hidden_dim': 128,
        'embedding_dim': 64,
        'epochs': 200,
        'learning_rate': 0.01,
        'eval_every': 20,
        'save_every': 100
    }
    
    print("⚙️  Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Step 1: Load and prepare dataset
    print(f"\n{'='*20} STEP 1: DATA PREPARATION {'='*20}")
    dataset = TranslationDataset(
        max_vocab_size=config['max_vocab_size'],
        max_seq_len=config['max_seq_len']
    )
    
    train_pairs, val_pairs, test_pairs = dataset.load_data()
    
    if not train_pairs:
        print("❌ No training data available. Exiting.")
        return
    
    # Step 2: Initialize model
    print(f"\n{'='*20} STEP 2: MODEL INITIALIZATION {'='*20}")
    
    # Model dimensions
    input_dim = len(dataset.src_vocab)   # Source vocabulary size
    output_dim = len(dataset.tgt_vocab)  # Target vocabulary size  
    hidden_dim = config['hidden_dim']
    vocab_size = len(dataset.tgt_vocab)  # Output vocabulary size
    
    print(f"📐 Model Architecture:")
    print(f"   Input dimension (source vocab):  {input_dim}")
    print(f"   Output dimension (target vocab): {output_dim}")
    print(f"   Hidden dimension:                {hidden_dim}")
    print(f"   Vocabulary size:                 {vocab_size}")
    
    model = GRUEncoderDecoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        vocab_size=vocab_size,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    )
    
    print("✅ Model initialized successfully!")
    
    # Step 3: Create trainer and start training
    print(f"\n{'='*20} STEP 3: TRAINING {'='*20}")
    
    trainer = TranslationTrainer(dataset, model)
    
    # Train the model
    history = trainer.train(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        eval_every=config['eval_every'],
        save_every=config['save_every']
    )
    
    # Step 4: Final evaluation and testing
    print(f"\n{'='*20} STEP 4: FINAL EVALUATION {'='*20}")
    
    # Final evaluation
    if val_pairs:
        final_val_loss, final_val_bleu = trainer.evaluate(val_pairs, max_samples=100)
        print(f"🎯 Final Validation Loss: {final_val_loss:.4f}")
        print(f"🎯 Final Validation BLEU: {final_val_bleu:.4f}")
    
    if test_pairs:
        final_test_loss, final_test_bleu = trainer.evaluate(test_pairs, max_samples=100)
        print(f"🎯 Final Test Loss: {final_test_loss:.4f}")
        print(f"🎯 Final Test BLEU: {final_test_bleu:.4f}")
    
    # Step 5: Test translations
    print(f"\n{'='*20} STEP 5: TRANSLATION EXAMPLES {'='*20}")
    
    if test_pairs:
        trainer.test_translation(test_pairs, num_examples=10)
    elif val_pairs:
        trainer.test_translation(val_pairs, num_examples=10)
    else:
        trainer.test_translation(train_pairs[:10], num_examples=10)
    
    # Step 6: Save final model and plot results
    print(f"\n{'='*20} STEP 6: SAVING AND VISUALIZATION {'='*20}")
    
    trainer.save_model('final_translation_model.pkl')
    
    # Plot training curves
    try:
        plot_training_history(history)
    except Exception as e:
        print(f"⚠️  Could not plot training curves: {e}")
    
    # Summary
    print(f"\n{'='*20} TRAINING SUMMARY {'='*20}")
    print(f"✅ Training completed successfully!")
    print(f"📊 Total epochs: {config['epochs']}")
    print(f"📚 Training samples used: {min(1000, len(train_pairs))}")
    print(f"💾 Model saved as: final_translation_model.pkl")
    
    if history['val_loss']:
        best_val_loss = min(history['val_loss'])
        best_val_epoch = history['epochs'][np.argmin(history['val_loss'])]
        print(f"🏆 Best validation loss: {best_val_loss:.4f} (epoch {best_val_epoch})")
    
    if history['val_bleu']:
        best_val_bleu = max(history['val_bleu'])
        best_bleu_epoch = history['epochs'][np.argmax(history['val_bleu'])]
        print(f"🏆 Best validation BLEU: {best_val_bleu:.4f} (epoch {best_bleu_epoch})")
    
    print("=" * 80)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    main()
