from learning import Trainer

trainer = Trainer()

trainer.train(epochs=5, batch_size=32, lr=0.025, min_lr=0.001, window_size=12, negative_samples=6, saves_per_epoch=30)
