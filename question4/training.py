import sys
import argparse
import time
import datetime
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import EPARNNModel, EPADNNModel
from dsets import EpdDataset
from logconf import logging

from torch.utils.tensorboard import SummaryWriter

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def enumerateWithEstimate(
        iter,
        desc_str,
        start_ndx=0,
        print_ndx=4,
        backoff=None,
        iter_len=None,
):
    if iter_len is None:
        iter_len = len(iter)

    if backoff is None:
        backoff = 2
        while backoff ** 7 < iter_len:
            backoff *= 2

    assert backoff >= 2
    while print_ndx < start_ndx * backoff:
        print_ndx *= backoff

    log.warning("{} ----/{}, starting".format(
        desc_str,
        iter_len,
    ))
    start_ts = time.time()
    for (current_ndx, item) in enumerate(iter):
        yield (current_ndx, item)

        if current_ndx == print_ndx:
            # ... <1>
            duration_sec = ((time.time() - start_ts)
                            / (current_ndx - start_ndx + 1)
                            * (iter_len-start_ndx)
                            )

            done_dt = datetime.datetime.fromtimestamp(start_ts + duration_sec)
            done_td = datetime.timedelta(seconds=duration_sec)

            log.info("{} {:-4}/{}, done at {}, {}".format(
                desc_str,
                current_ndx,
                iter_len,
                str(done_dt).rsplit('.', 1)[0],
                str(done_td).rsplit('.', 1)[0],
            ))

            print_ndx *= backoff

        if current_ndx + 1 == start_ndx:
            start_ts = time.time()

    log.warning("{} ----/{}, done at {}".format(
        desc_str,
        iter_len,
        str(datetime.datetime.now()).rsplit('.', 1)[0],
    ))


class EPATrainingApp:
    _METRICS_R2_NDX = 0
    _METRICS_LOSS_NDX = 1
    _METRICS_SIZE = 2

    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()

        parser.add_argument('--num-workers',
                            help='Number of worker processes for background data loading',
                            default=0,
                            type=int,
                            )

        parser.add_argument('--batch-size',
                            help='Batch size to use for training',
                            default=32,
                            type=int,
                            )

        parser.add_argument('--epochs',
                            help='Number of epochs to train for',
                            default=1,
                            type=int,
                            )

        parser.add_argument('--tb-prefix',
                            default='epa',
                            help="Data prefix to use for Tensorboard run. Defaults to epa.",
                            )

        parser.add_argument('comment',
                            help="Comment suffix for Tensorboard run.",
                            nargs='?',
                            default='epa',
                            )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join(
                'runs', self.cli_args.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '-trn_cls-' + self.cli_args.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_cls-' + self.cli_args.comment)

    def initModel(self):
        model = EPADNNModel()

        if self.use_cuda:
            log.info("Using CUDA; {} device(s)".format(
                torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)

        return model

    def initOptimizer(self):
        return optim.Adam(
            self.model.parameters(),
            lr=0.001,
            # momentum=0.99
        )

    def initTrainDl(self):
        train_ds = EpdDataset(
            is_val=False,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return train_dl

    def initValDl(self):
        val_ds = EpdDataset(
            is_val=True,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return val_dl

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            log.info(
                "Epoch {} of {}, {}/{} batches of size {}*{}".format(
                    epoch_ndx,
                    self.cli_args.epochs,
                    len(train_dl),
                    len(val_dl),
                    self.cli_args.batch_size,
                    (torch.cuda.device_count() if self.use_cuda else 1),
                )
            )

            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            valMetrics_t = self.doValidation(epoch_ndx, val_dl)
            self.logMetrics(epoch_ndx, 'val', valMetrics_t)

    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()

        trnMetrics_g = torch.zeros(
            EPATrainingApp._METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
        )

        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )

        for batch_ndx, batch_tup in batch_iter:

            self.optimizer.zero_grad()

            loss_var = self.computeBatchLoss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trnMetrics_g
            )

            loss_var.backward()
            self.optimizer.step()

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()

            valMetrics_g = torch.zeros(
                EPATrainingApp._METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(
                    batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')

    def computeR2(self, test_t, pred_t):
        res_t = torch.zeros(test_t.size(0)).to(device=self.device)
        for i in range(test_t.size(0)):
            res_t[i] = 1 - ((test_t[i] - pred_t[i]) ** 2).sum() / \
                ((test_t[i] - test_t[i].mean()) ** 2).sum()
        return res_t

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, output_t = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        output_g = output_t.to(self.device, non_blocking=True)

        pred_g = self.model(input_g)

        loss_func = nn.MSELoss(reduction='none')

        loss_g = loss_func(
            pred_g,
            output_g,
        )

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + output_t.size(0)

        metrics_g[EPATrainingApp._METRICS_R2_NDX, start_ndx:end_ndx] = self.computeR2(
            output_g.detach(), pred_g.detach())
        metrics_g[EPATrainingApp._METRICS_LOSS_NDX, start_ndx:end_ndx] = \
            loss_g[:, 0].detach()
        return loss_g.mean()

    def logMetrics(
        self,
        epoch_ndx,
        mode_str,
        metrics_t
    ):
        self.initTensorboardWriters()

        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        metrics_dict = {}

        metrics_dict['loss/all'] = \
            metrics_t[EPATrainingApp._METRICS_LOSS_NDX].mean()
        metrics_dict['r2/all'] = \
            metrics_t[EPATrainingApp._METRICS_R2_NDX].mean()

        log.info(
            "E{} {:8} {loss/all:.4f} loss".format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )

        log.info(
            "E{} {:8} {r2/all:.4f} loss".format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )

        # rss_t = metrics_t[EPATrainingApp._METRICS_PRED_NDX] - \
        #     metrics_t[EPATrainingApp._METRICS_OUTPUT_NDX]
        # rss_t = rss_t * rss_t

        # tss_t = metrics_t[EPATrainingApp._METRICS_OUTPUT_NDX].mean(
        # ) - metrics_t[EPATrainingApp._METRICS_OUTPUT_NDX]
        # tss_t = tss_t * tss_t

        # metrics_dict['rs/all'] = 1 - float(rss_t.sum()) / float(tss_t.sum())

        # log.info(
        #     "E{} {:8} {rs/all:.4f} std".format(
        #         epoch_ndx,
        #         mode_str,
        #         **metrics_dict,
        #     )
        # )

        writer = getattr(self, mode_str + '_writer')

        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTrainingSamples_count)
