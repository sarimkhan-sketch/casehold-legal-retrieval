"""Extension 1: fine-tune all-MiniLM-L6-v2 on CaseHOLD training pairs.

Each example is (citing context, correct holding, one wrong option). The
multiple negatives ranking loss treats the other holdings in the batch as
extra negatives. One epoch, batch 32, learning rate 2e-5.
"""
import time
from pathlib import Path

from datasets import Dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments, losses)
from sentence_transformers.training_args import BatchSamplers

from casehold_ir.data import load_train_pairs
from casehold_ir.models import device

BASE = "sentence-transformers/all-MiniLM-L6-v2"
OUT = Path(__file__).resolve().parent.parent / "models" / "minilm-casehold"


def main():
    anchors, positives, negatives = load_train_pairs()
    train = Dataset.from_dict({"anchor": anchors, "positive": positives, "negative": negatives})
    model = SentenceTransformer(BASE, device=device())
    args = SentenceTransformerTrainingArguments(
        output_dir=str(OUT / "checkpoints"), num_train_epochs=1,
        per_device_train_batch_size=32, learning_rate=2e-5, warmup_ratio=0.1,
        batch_sampler=BatchSamplers.NO_DUPLICATES, seed=13,
        logging_steps=100, save_strategy="no", report_to="none",
        dataloader_pin_memory=False)
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train,
                                         loss=losses.MultipleNegativesRankingLoss(model))
    start = time.time()
    trainer.train()
    print(f"training took {(time.time() - start) / 60:.1f} minutes")
    model.save(str(OUT))


if __name__ == "__main__":
    main()
