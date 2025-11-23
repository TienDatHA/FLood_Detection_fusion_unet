import tensorflow as tf

# Cấu hình TensorFlow để tối ưu hóa memory usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Cho phép memory growth để tránh OOM
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Giới hạn memory usage nếu cần
        # tf.config.experimental.set_virtual_device_configuration(
        #     gpus[0],
        #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*8)]
        # )
    except RuntimeError as e:
        print(e)

tf.version.VERSION
from tensorflow.keras.layers import Input
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from pathlib import Path
import os
import datetime
import json
import numpy as np
from utils import load_data, Cust_DatasetGenerator, Inference
from model import Resnet50_UNet
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()
from config import *

class SmartEarlyStopping(keras.callbacks.Callback):
    """
    Early Stopping thông minh để tránh overfitting
    Theo dõi cả validation loss và gap giữa train/val metrics
    """
    def __init__(self, 
                 monitor_val_loss=True,
                 monitor_overfitting=True,
                 patience_val_loss=15,
                 patience_overfitting=10,
                 min_delta_val_loss=0.001,
                 overfitting_threshold=0.1,
                 restore_best_weights=True,
                 verbose=1):
        super().__init__()
        
        self.monitor_val_loss = monitor_val_loss
        self.monitor_overfitting = monitor_overfitting
        self.patience_val_loss = patience_val_loss
        self.patience_overfitting = patience_overfitting
        self.min_delta_val_loss = min_delta_val_loss
        self.overfitting_threshold = overfitting_threshold
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        # Tracking variables
        self.best_val_loss = float('inf')
        self.best_weights = None
        self.wait_val_loss = 0
        self.wait_overfitting = 0
        self.best_epoch = 0
        
        # Overfitting tracking
        self.train_losses = []
        self.val_losses = []
        
    def on_train_begin(self, logs=None):
        self.best_val_loss = float('inf')
        self.wait_val_loss = 0
        self.wait_overfitting = 0
        self.best_epoch = 0
        self.train_losses = []
        self.val_losses = []
        
        if self.verbose > 0:
            print(f"\n🛡️  Smart Early Stopping activated:")
            if self.monitor_val_loss:
                print(f"   ├─ Validation Loss patience: {self.patience_val_loss} epochs")
            if self.monitor_overfitting:
                print(f"   └─ Overfitting patience: {self.patience_overfitting} epochs")
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        current_val_loss = logs.get('val_loss', float('inf'))
        current_train_loss = logs.get('loss', float('inf'))
        
        # Store losses for overfitting analysis
        self.train_losses.append(current_train_loss)
        self.val_losses.append(current_val_loss)
        
        # Check validation loss improvement
        val_loss_improved = False
        if self.monitor_val_loss:
            if current_val_loss < self.best_val_loss - self.min_delta_val_loss:
                self.best_val_loss = current_val_loss
                self.best_epoch = epoch + 1
                self.wait_val_loss = 0
                val_loss_improved = True
                
                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()
                    
                if self.verbose > 0:
                    print(f"🎯 Val loss improved to {self.best_val_loss:.4f} (epoch {self.best_epoch})")
            else:
                self.wait_val_loss += 1
                if self.verbose > 1:
                    print(f"⏳ Val loss plateau: {self.wait_val_loss}/{self.patience_val_loss}")
        
        # Check for overfitting
        overfitting_detected = False
        if self.monitor_overfitting and len(self.train_losses) >= 5:
            # Calculate recent trend (last 5 epochs)
            recent_train = np.mean(self.train_losses[-5:])
            recent_val = np.mean(self.val_losses[-5:])
            
            # Check if validation loss is significantly higher than training loss
            if recent_val > recent_train + self.overfitting_threshold:
                overfitting_detected = True
                self.wait_overfitting += 1
                if self.verbose > 1:
                    gap = recent_val - recent_train
                    print(f"⚠️  Overfitting detected: train/val gap = {gap:.4f} ({self.wait_overfitting}/{self.patience_overfitting})")
            else:
                self.wait_overfitting = 0
        
        # Decision to stop
        should_stop = False
        stop_reason = ""
        
        if self.monitor_val_loss and self.wait_val_loss >= self.patience_val_loss:
            should_stop = True
            stop_reason += f"Val loss plateau ({self.patience_val_loss} epochs)"
        
        if self.monitor_overfitting and self.wait_overfitting >= self.patience_overfitting:
            should_stop = True
            if stop_reason:
                stop_reason += " + "
            stop_reason += f"Overfitting detected ({self.patience_overfitting} epochs)"
        
        if should_stop:
            if self.verbose > 0:
                print(f"\n🛑 Early stopping triggered: {stop_reason}")
                print(f"🏆 Best val_loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
                
                if self.restore_best_weights and self.best_weights is not None:
                    print("🔄 Restoring best weights...")
                    self.model.set_weights(self.best_weights)
            
            self.model.stop_training = True

class BestModelLogger(keras.callbacks.Callback):
    """Custom callback để lưu checkpoint tốt nhất và log metrics"""
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Khởi tạo các metrics tốt nhất
        self.best_val_loss = float('inf')
        self.best_iou = 0.0
        self.best_performance = 0.0  # Combination of IoU and F1
        
        # Epochs tương ứng
        self.best_val_loss_epoch = 0
        self.best_iou_epoch = 0
        self.best_performance_epoch = 0
        
        # Log file
        self.log_file = self.log_dir / "training_log.txt"
        
        # Tạo log file và ghi header
        with open(self.log_file, 'w') as f:
            f.write("=== FLOOD DETECTION TRAINING LOG ===\n")
            f.write(f"Training started at: {datetime.datetime.now()}\n")
            f.write("="*50 + "\n\n")
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        current_val_loss = logs.get('val_loss', float('inf'))
        current_iou = logs.get('val_iou_score', 0.0)
        current_f1 = logs.get('val_f1-score', 0.0)
        current_performance = (current_iou + current_f1) / 2  # Combined metric
        
        # Update best val_loss
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.best_val_loss_epoch = epoch + 1
            self.model.save_weights(self.log_dir / "best_val_loss_checkpoint.h5")
            print(f"\n🎯 New best val_loss: {self.best_val_loss:.4f} at epoch {self.best_val_loss_epoch}")
        
        # Update best IoU
        if current_iou > self.best_iou:
            self.best_iou = current_iou
            self.best_iou_epoch = epoch + 1
            self.model.save_weights(self.log_dir / "best_iou_checkpoint.h5")
            print(f"\n🎯 New best IoU: {self.best_iou:.4f} at epoch {self.best_iou_epoch}")
        
        # Update best performance
        if current_performance > self.best_performance:
            self.best_performance = current_performance
            self.best_performance_epoch = epoch + 1
            self.model.save_weights(self.log_dir / "best_performance_checkpoint.h5")
            print(f"\n🎯 New best performance: {self.best_performance:.4f} at epoch {self.best_performance_epoch}")
        
        # Log to file
        with open(self.log_file, 'a') as f:
            f.write(f"Epoch {epoch + 1:3d} | ")
            f.write(f"Loss: {logs.get('loss', 0):.4f} | ")
            f.write(f"Val_Loss: {current_val_loss:.4f} | ")
            f.write(f"IoU: {current_iou:.4f} | ")
            f.write(f"F1: {current_f1:.4f} | ")
            f.write(f"Performance: {current_performance:.4f}\n")
    
    def on_train_end(self, logs=None):
        """Ghi tổng kết khi training kết thúc"""
        with open(self.log_file, 'a') as f:
            f.write("\n" + "="*50 + "\n")
            f.write("TRAINING COMPLETED SUMMARY\n")
            f.write("="*50 + "\n")
            f.write(f"Training ended at: {datetime.datetime.now()}\n\n")
            
            f.write("🏆 BEST RESULTS:\n")
            f.write(f"├─ Best Val Loss: {self.best_val_loss:.4f} at epoch {self.best_val_loss_epoch}\n")
            f.write(f"├─ Best IoU:      {self.best_iou:.4f} at epoch {self.best_iou_epoch}\n")
            f.write(f"└─ Best Performance: {self.best_performance:.4f} at epoch {self.best_performance_epoch}\n\n")
            
            f.write("💾 SAVED CHECKPOINTS:\n")
            f.write(f"├─ best_val_loss_checkpoint.h5 (epoch {self.best_val_loss_epoch})\n")
            f.write(f"├─ best_iou_checkpoint.h5 (epoch {self.best_iou_epoch})\n")
            f.write(f"└─ best_performance_checkpoint.h5 (epoch {self.best_performance_epoch})\n")
        
        print(f"\n🎉 Training completed! Check logs at: {self.log_file}")

def setup_tensorboard_and_callbacks(log_dir="logs", use_early_stopping=True):
    """Thiết lập TensorBoard và các callbacks"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # TensorBoard callback
    tensorboard_log_dir = log_dir / f"tensorboard_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tensorboard = TensorBoard(
        log_dir=str(tensorboard_log_dir),
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch'
    )
    
    # Best model logger
    best_logger = BestModelLogger(log_dir)
    
    callbacks = [tensorboard, best_logger]
    
    # Smart Early Stopping (có thể tắt nếu không muốn)
    if use_early_stopping:
        smart_early_stopping = SmartEarlyStopping(
            monitor_val_loss=True,
            monitor_overfitting=True,
            patience_val_loss=15,
            patience_overfitting=10,
            min_delta_val_loss=0.001,
            overfitting_threshold=0.1,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(smart_early_stopping)
    
    # Standard model checkpoint (fallback)
    checkpointer = ModelCheckpoint(
        str(log_dir / 'standard_checkpoint.h5'),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True
    )
    callbacks.append(checkpointer)
    
    return callbacks

def log_evaluation_results(model, val_x, val_y, log_dir="logs"):
    """Hàm xuất log khi evaluate model"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    eval_log_file = log_dir / "evaluation_log.txt"
    
    print("🔍 Starting model evaluation...")
    
    # Tính toán metrics
    intersection, union = 0, 0
    all_predictions = []
    all_targets = []
    
    for ind in range(len(val_x)):
        ints, un = Inference(ind, val_x, val_y, model)
        intersection += ints
        union += un
    
    iou = intersection / union if union > 0 else 0
    
    # Ghi log evaluation
    with open(eval_log_file, 'w') as f:
        f.write("=== FLOOD DETECTION EVALUATION LOG ===\n")
        f.write(f"Evaluation performed at: {datetime.datetime.now()}\n")
        f.write("="*50 + "\n\n")
        
        f.write("📊 EVALUATION METRICS:\n")
        f.write(f"├─ IoU Score: {iou:.4f}\n")
        f.write(f"├─ Total Intersection: {intersection}\n")
        f.write(f"├─ Total Union: {union}\n")
        f.write(f"└─ Number of samples evaluated: {len(val_x)}\n\n")
        
        f.write("📁 PREDICTION OUTPUTS:\n")
        f.write(f"└─ Saved to: {WEIGHT_PATH / 'Pred_Mask'}\n")
    
    print(f"📈 IoU Score: {iou:.4f}")
    print(f"📝 Evaluation log saved to: {eval_log_file}")
    
    return iou

def train_fusion(use_early_stopping=True):
    print("🚀 Starting Fusion U-Net Training...")
    
    # Thiết lập logs và callbacks
    log_dir = Path("training_logs")
    callbacks = setup_tensorboard_and_callbacks(log_dir, use_early_stopping)
    
    print(f"📊 TensorBoard logs will be saved to: {log_dir}")
    print(f"📝 Training logs will be saved to: {log_dir}/training_log.txt")
    print(f"💾 Best checkpoints will be saved to: {log_dir}/")
    
    if use_early_stopping:
        print("🛡️  Smart Early Stopping is ENABLED")
    else:
        print("⚠️  Smart Early Stopping is DISABLED - will run full epochs")
    
    model = Resnet50_UNet(n_classes, in_img, in_inf)
    model.compile(optimizer, loss = total_loss, metrics = metrics)
    
    # Phase 1: Freeze and tune (only decoder layers)
    print("\n🔒 Phase 1: Training with frozen encoder (2 epochs)...")
    for layer in model.layers:  
        if 'DEC_' not in layer.name:
            layer.trainable = False
    
    # Log trainable parameters
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    total_params = model.count_params()
    print(f"📊 Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    model.fit(
        my_training_batch_generator, 
        validation_data=my_validation_batch_generator,  
        epochs=2,  
        steps_per_epoch=int(len(train_x)/train_batchSize), 
        validation_steps=int(len(val_x)/val_batchSize),
        callbacks=[scheduler] + callbacks
    )
    
    # Phase 2: Unfreeze and train all layers
    print("\n🔓 Phase 2: Training with all layers unfrozen (100 epochs)...")
    for layer in model.layers:
        layer.trainable = True
    
    # Log trainable parameters after unfreezing
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    print(f"📊 Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    model.fit(
        my_training_batch_generator, 
        validation_data=my_validation_batch_generator,  
        epochs=100,  
        steps_per_epoch=int(len(train_x)/train_batchSize), 
        validation_steps=int(len(val_x)/val_batchSize),
        callbacks=[scheduler] + callbacks
    )
    
    print("\n✅ Training completed successfully!")
    print(f"🔍 To view TensorBoard: tensorboard --logdir {log_dir}")
    print(f"📖 Check training summary in: {log_dir}/training_log.txt")

def evaluate_fusion():
    print("🔍 Starting Fusion U-Net Evaluation...")
    
    model = Resnet50_UNet(n_classes, in_img, in_inf)
    
    # Kiểm tra xem có checkpoint nào tốt nhất không
    log_dir = Path("training_logs")
    best_checkpoints = {
        "best_performance": log_dir / "best_performance_checkpoint.h5",
        "best_iou": log_dir / "best_iou_checkpoint.h5", 
        "best_val_loss": log_dir / "best_val_loss_checkpoint.h5",
        "standard": WEIGHT_PATH / WEIGHT_FILE
    }
    
    # Tìm checkpoint tốt nhất có sẵn
    checkpoint_to_use = None
    checkpoint_name = ""
    
    for name, path in best_checkpoints.items():
        if path.exists():
            checkpoint_to_use = path
            checkpoint_name = name
            break
    
    if checkpoint_to_use is None:
        print("❌ No checkpoint found!")
        return
    
    print(f"📥 Loading checkpoint: {checkpoint_name} ({checkpoint_to_use})")
    model.load_weights(str(checkpoint_to_use))
    
    # Tạo thư mục output
    OUT_FOLDER = WEIGHT_PATH / 'Pred_Mask'
    if not os.path.exists(OUT_FOLDER): 
        os.mkdir(OUT_FOLDER)
    
    print(f"💾 Predictions will be saved to: {OUT_FOLDER}")
    
    # Đánh giá model và log kết quả
    file_x, file_y = val_x, val_y
    iou_score = log_evaluation_results(model, file_x, file_y, log_dir="evaluation_logs")
    
    print(f"\n📊 Final IoU Score: {iou_score:.4f}")
    print(f"✅ Evaluation completed!")
    
    return iou_score

if __name__ == '__main__':
    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train the network.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate'")
    parser.add_argument("--no-early-stopping", 
                        action="store_true",
                        help="Disable smart early stopping")
    parser.add_argument("--early-stopping-patience",
                        type=int,
                        default=15,
                        help="Patience for validation loss (default: 15)")
    parser.add_argument("--overfitting-patience",
                        type=int, 
                        default=10,
                        help="Patience for overfitting detection (default: 10)")
    args = parser.parse_args()
    
    # load data
    train_x, train_y, val_x, val_y = load_data()
    my_training_batch_generator = Cust_DatasetGenerator(train_x, train_y, batch_size=train_batchSize)
    my_validation_batch_generator = Cust_DatasetGenerator(val_x, val_y, batch_size=val_batchSize)
    
    # define network parameters
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    activation = 'sigmoid' if n_classes == 1 else 'softmax'

    in_img = Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))
    in_inf = Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))
    
    # define loss, optimizer, lr etc.
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
    total_loss = 0.2 * dice_loss + (0.8 * focal_loss)
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.00001)

    if args.command == "train":
        print('In training Fusion Network')
        use_early_stopping = not args.no_early_stopping
        train_fusion(use_early_stopping)
        
    if args.command == "evaluate":
        print('Evaluating Fusion Network')
        evaluate_fusion()
