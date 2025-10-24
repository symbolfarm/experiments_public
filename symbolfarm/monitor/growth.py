import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GrowthMonitor:
    """Monitor training progress and decide when to grow the network."""
    
    def __init__(self, patience: int, threshold: float, min_epochs: int = 1):
        self.patience = patience
        self.threshold = threshold
        self.min_epochs = min_epochs
        
        self.best_loss = float('inf')
        self.epochs_since_improvement = 0
        self.epoch_count = 0
        self.loss_history = []
    
    def update(self, val_loss: float) -> bool:
        """
        Update with validation loss and return whether to grow.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if network should grow, False otherwise
        """
        self.epoch_count += 1
        self.loss_history.append(val_loss)
        
        # Don't grow too early
        if self.epoch_count < self.min_epochs:
            return False
        
        # Check for improvement
        if val_loss < self.best_loss - self.threshold:
            self.best_loss = val_loss
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1
        
        # Decide whether to grow
        should_grow = self.epochs_since_improvement >= self.patience
        
        if should_grow:
            logger.info(f"Growth triggered: {self.epochs_since_improvement} epochs without improvement")
            self.reset()
        
        return should_grow
    
    def reset(self):
        """Reset after growth."""
        self.epochs_since_improvement = 0
        self.best_loss = float('inf')  # Reset to allow for potential temporary increase after growth

