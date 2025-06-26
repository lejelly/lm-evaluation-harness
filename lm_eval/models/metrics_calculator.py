#!/usr/bin/env python3
"""
Metrics calculation utilities for logits analysis
"""

import numpy as np
import pandas as pd
from pathlib import Path
import csv
import threading
import os

class MetricsCalculator:
    """
    Calculate various metrics from logits and save to CSV
    """
    
    def __init__(self, csv_path="generation_metrics.csv"):
        self.csv_path = Path(csv_path)
        self._lock = threading.Lock()
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Initialize CSV file with headers if it doesn't exist"""
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'instance_id',
                    'task_name', 
                    'doc_id',
                    'idx',
                    'model_name',
                    'vocab_size',
                    'sequence_length',
                    'shannon_entropy',
                    'is_correct',  # Will be filled later
                ])
    
    @staticmethod
    def compute_shannon_entropy(logits):
        """
        Shannon Entropyを計算（全トークンの平均）
        
        Args:
            logits: [sequence_length, vocab_size] のlogits
            
        Returns:
            float: Shannon Entropy（全トークンの平均）
        """
        entropies = []
        
        # 各トークンのエントロピーを計算
        for step_logits in logits:  # [vocab_size]
            # float64に変換して数値精度を向上
            step_logits = step_logits.astype(np.float32)
            
            # softmaxで確率分布に変換（数値安定版）
            logits_max = np.max(step_logits)
            exp_logits = np.exp(step_logits - logits_max)
            probs = exp_logits / np.sum(exp_logits)
            
            # Shannon Entropy計算: -sum(p * log(p))
            # 注: 自然対数を使用（情報理論の標準）
            # p=0の場合、0*log(0)=0として扱う
            mask = probs > 0
            if np.any(mask):
                step_entropy = -np.sum(probs[mask] * np.log(probs[mask]))
            else:
                step_entropy = 0.0
            
            # NaNチェック
            if np.isnan(step_entropy) or np.isinf(step_entropy):
                print(f"ERROR: Shannon Entropy calculation resulted in NaN/Inf!")
                print(f"  step_logits shape: {step_logits.shape}")
                print(f"  step_logits min/max: {np.min(step_logits)}/{np.max(step_logits)}")
                print(f"  probs sum: {np.sum(probs)}")
                print(f"  non-zero probs: {np.sum(mask)}")
                step_entropy = -1.0
                
            entropies.append(step_entropy)
        
        # 全トークンの平均を返す
        return np.mean(entropies)
    
    
    def calculate_and_save_metrics(self, logits_sequence, context_tokens, generated_tokens, 
                                   tokenizer, model_name, instance_metadata=None):
        """
        Calculate metrics and save to CSV
        
        Args:
            logits_sequence: List of logits tensors
            context_tokens: Context tokens
            generated_tokens: Generated tokens
            tokenizer: Tokenizer instance
            model_name: Model name
            instance_metadata: Instance metadata dict
        """
        try:
            # Convert to numpy array
            logits_array = np.array(logits_sequence)
            
            # Handle different input shapes
            if len(logits_array.shape) == 2:
                # Already in [steps, vocab_size] format (from batch processing)
                logits = logits_array
            elif len(logits_array.shape) == 3:
                # [steps, batch_size, vocab_size] format
                if logits_array.shape[1] != 1:
                    print(f"Warning: Expected batch_size=1, got {logits_array.shape[1]}")
                    return
                logits = logits_array[:, 0, :]  # [steps, vocab_size]
            else:
                print(f"Error: Unexpected logits shape: {logits_array.shape}")
                return
            
            # Shannon Entropyを計算
            vocab_size = len(tokenizer)
            shannon_entropy = self.compute_shannon_entropy(logits)
            
            # Instance IDを作成
            import hashlib
            import time
            context_text = tokenizer.decode(context_tokens[0], skip_special_tokens=True)
            context_hash = hashlib.md5(context_text.encode()).hexdigest()[:8]
            timestamp = int(time.time() * 1000000)  # microsecond precision
            instance_id = f"{context_hash}_{timestamp}"
            
            # メタデータから値を取得
            task_name = instance_metadata.get('task_name', 'unknown') if instance_metadata else 'unknown'
            doc_id = instance_metadata.get('doc_id', -1) if instance_metadata else -1
            idx = instance_metadata.get('idx', -1) if instance_metadata else -1
            
            # デバッグ出力
            print(f"[Metrics Save] task_name={task_name}, doc_id={doc_id}, idx={idx}")
            
            # CSVに追記
            row = [
                instance_id,
                task_name,
                doc_id,
                idx,
                model_name,
                vocab_size,
                logits.shape[0],
                shannon_entropy,
                'pending',  # is_correct - will be filled later (use 'pending' instead of empty string)
            ]
            
            # Thread-safe CSV writing
            with self._lock:
                with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
            
            print(f"Metrics calculated and saved for instance: {instance_id}")
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
    
    def update_correctness(self, task_name, doc_id, idx, is_correct):
        """
        Update correctness information in CSV
        
        Args:
            task_name: Task name
            doc_id: Document ID
            idx: Instance index
            is_correct: Boolean indicating correctness
        """
        try:
            # Read existing CSV
            if not self.csv_path.exists():
                print(f"CSV file does not exist: {self.csv_path}")
                return
                
            # Read CSV with explicit dtype for is_correct column
            df = pd.read_csv(self.csv_path, dtype={'is_correct': str})
            
            # デバッグ出力
            print(f"[Correctness Update] Looking for: task_name={task_name}, doc_id={doc_id}, idx={idx}")
            
            # Find matching row - try exact match first, then fallback to doc_id only
            mask = (df['task_name'] == task_name) & (df['doc_id'] == doc_id) & (df['idx'] == idx)
            
            if not mask.any():
                # Fallback: match by task_name and doc_id only (for cases where idx is always 0 in CSV)
                mask = (df['task_name'] == task_name) & (df['doc_id'] == doc_id)
                if mask.any():
                    matched_idx = df[mask]['idx'].iloc[0]
                    print(f"[Correctness Update] Using fallback matching for {task_name}_{doc_id} (idx mismatch: expected {idx}, found {matched_idx})")
            
            if mask.any():
                # Convert is_correct column to string type first to avoid dtype issues
                df['is_correct'] = df['is_correct'].astype(str)
                # Update the value
                df.loc[mask, 'is_correct'] = str(is_correct)
                
                # Save back to CSV
                with self._lock:
                    df.to_csv(self.csv_path, index=False)
                    
                print(f"[Correctness Update] SUCCESS: Updated correctness for {task_name}_{doc_id}_{idx}: {is_correct}")
            else:
                print(f"No matching row found for {task_name}_{doc_id}_{idx}")
                # Debug: print first few rows to see what we have
                if len(df) > 0:
                    print(f"Available rows sample:")
                    print(df[['task_name', 'doc_id', 'idx']].head())
            
        except Exception as e:
            print(f"Error updating correctness: {e}")
            import traceback
            traceback.print_exc()