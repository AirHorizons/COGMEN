import copy
import time

import numpy as np
from numpy.core import overrides
import torch
from tqdm import tqdm
from sklearn import metrics

import cogmen



class Coach:
    def __init__(self, trainset, devset, testset, model, opt, sched, args, log, run):
        self.trainset = trainset
        self.devset = devset
        self.testset = testset
        self.model = model
        self.opt = opt
        self.scheduler = sched
        self.args = args
        self.dataset_label_dict = {
            "iemocap": {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5},
            "iemocap_4": {"hap": 0, "sad": 1, "neu": 2, "ang": 3},
            "mosei": {"Negative": 0, "Positive": 1},
        }

        if args.dataset and args.emotion == "multilabel":
            self.dataset_label_dict["mosei"] = {
                "happiness": 0,
                "sadness": 1,
                "anger": 2,
                "surprise": 3,
                "disgust": 4,
                "fear": 5,
            }

        if args.emotion == "7class":
            self.label_to_idx = {
                "Strong Negative": 0,
                "Weak Negative": 1,
                "Negative": 2,
                "Neutral": 3,
                "Positive": 4,
                "Weak Positive": 5,
                "Strong Positive": 6,
            }
        else:
            self.label_to_idx = self.dataset_label_dict[args.dataset]

        self.best_dev_f1 = None
        self.best_epoch = None
        self.best_report = None
        self.best_state = None

        self.log = log
        self.run = run

    def load_ckpt(self, ckpt):
        self.best_dev_f1 = ckpt["best_dev_f1"]
        self.best_epoch = ckpt["best_epoch"]
        self.best_state = ckpt["best_state"]
        self.model.load_state_dict(self.best_state)
        print("Loaded best model.....")

    def train(self):
        self.log.debug(self.model)
        # Early stopping.
        best_dev_f1, best_epoch, best_state, best_report = (
            self.best_dev_f1,
            self.best_epoch,
            self.best_state,
            self.best_report
        )

        dev_f1s = []
        test_f1s = []
        train_losses = []
        best_test_f1 = None

        # Train
        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.train_epoch(epoch)
            report, dev_f1, dev_loss = self.evaluate()
            self.scheduler.step(dev_loss)
            report, test_f1, _ = self.evaluate(test=True)
            if self.args.dataset == "mosei" and self.args.emotion == "multilabel":
                test_f1 = np.array(list(test_f1.values())).mean()
            self.log.info("[Dev set] [f1 {:.4f}]".format(dev_f1))
            if best_dev_f1 is None or dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_test_f1 = test_f1
                best_report = report
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                if self.args.dataset == "mosei":
                    torch.save(
                        {"args": self.args, "state_dict": self.model},
                        "model_checkpoints/mosei_best_dev_f1_model_"
                        + self.args.modalities
                        + "_"
                        + self.args.emotion
                        + ".pt",
                    )
                else:
                    torch.save(
                        {"args": self.args, "state_dict": self.model},
                        "model_checkpoints/"
                        + self.args.dataset
                        + "_best_dev_f1_model_"
                        + self.args.modalities
                        + ".pt",
                    )

                self.log.info("Save the best model.")
            self.log.info("[Test set] [f1 {:.4f}]".format(test_f1))

            dev_f1s.append(dev_f1)
            test_f1s.append(test_f1)
            train_losses.append(train_loss)
            if self.args.wandb:
                self.run.log({
                    "f1 (dev)": dev_f1,
                    "f1 (test)": test_f1,
                    "train_loss": train_loss,
                    "val_loss": dev_loss,
                    }
                )
                for label in self.dataset_label_dict[self.args.dataset]:
                    for metric, value in report[label].items():
                        self.run.log({f"{label}/{metric}": value})
        if self.args.wandb:
            self.run.log({
                "best f1 (dev)": best_dev_f1,
                "best f1 (test)": best_test_f1,
            })

            return best_dev_f1, best_epoch, best_state, train_losses, dev_f1s, test_f1s

        # The best

        self.model.load_state_dict(best_state)
        self.log.info("")
        self.log.info("Best in epoch {}:".format(best_epoch))
        _, dev_f1, _ = self.evaluate()
        self.log.info("[Dev set] [f1 {:.4f}]".format(dev_f1))
        _, test_f1, _ = self.evaluate(test=True)
        self.log.info("[Test set] f1 {}".format(test_f1))

        for label in self.dataset_label_dict[self.args.dataset]:
            for metric, value in best_report[label].items():
                if self.args.wandb:
                    self.run.log({f"best_{label}/{metric}": value})
                self.log.info(f"best_{label}/{metric}: {value}")

        return best_dev_f1, best_epoch, best_state, train_losses, dev_f1s, test_f1s

    def train_epoch(self, epoch):
        start_time = time.time()
        epoch_loss = 0
        self.model.train()

        self.trainset.shuffle()
        for idx in tqdm(range(len(self.trainset)), desc="train epoch {}".format(epoch)):
            self.model.zero_grad()
            data = self.trainset[idx]
            for k, v in data.items():
                if not k == "utterance_texts":
                    data[k] = v.to(self.args.device)

            nll = self.model.get_loss(data)
            epoch_loss += nll.item()
            nll.backward()
            self.opt.step()

        end_time = time.time()
        self.log.info("")
        self.log.info(
            "[Epoch %d] [Loss: %f] [Time: %f]"
            % (epoch, epoch_loss, end_time - start_time)
        )
        return epoch_loss

    def evaluate(self, test=False):
        dev_loss = 0
        dataset = self.testset if test else self.devset
        self.model.eval()
        report = None
        with torch.no_grad():
            golds = []
            preds = []
            for idx in tqdm(range(len(dataset)), desc="test" if test else "dev"):
                data = dataset[idx]
                golds.append(data["label_tensor"])
                for k, v in data.items():
                    if not k == "utterance_texts":
                        data[k] = v.to(self.args.device)
                y_hat = self.model(data)
                preds.append(y_hat.detach().to("cpu"))
                nll = self.model.get_loss(data)
                dev_loss += nll.item()

            if self.args.dataset == "mosei" and self.args.emotion == "multilabel":
                golds = torch.cat(golds, dim=0).numpy()
                preds = torch.cat(preds, dim=0).numpy()
                f1 = metrics.f1_score(golds, preds, average="weighted")
                acc = metrics.accuracy_score(golds, preds)
            else:
                golds = torch.cat(golds, dim=-1).numpy()
                preds = torch.cat(preds, dim=-1).numpy()
                f1 = metrics.f1_score(golds, preds, average="weighted")

            if test:
                report = metrics.classification_report(
                        golds, preds, target_names=self.label_to_idx.keys(), digits=4, output_dict = True, zero_division = 0
                    )
                self.log.info(str(report))

                if self.args.dataset == "mosei" and self.args.emotion == "multilabel":
                    happy = metrics.f1_score(
                        golds[:, 0], preds[:, 0], average="weighted"
                    )
                    sad = metrics.f1_score(golds[:, 1], preds[:, 1], average="weighted")
                    anger = metrics.f1_score(
                        golds[:, 2], preds[:, 2], average="weighted"
                    )
                    surprise = metrics.f1_score(
                        golds[:, 3], preds[:, 3], average="weighted"
                    )
                    disgust = metrics.f1_score(
                        golds[:, 4], preds[:, 4], average="weighted"
                    )
                    fear = metrics.f1_score(
                        golds[:, 5], preds[:, 5], average="weighted"
                    )

                    f1 = {
                        "happy": happy,
                        "sad": sad,
                        "anger": anger,
                        "surprise": surprise,
                        "disgust": disgust,
                        "fear": fear,
                    }

        return report, f1, dev_loss
