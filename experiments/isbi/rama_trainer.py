import torch
from torch_em.trainer import DefaultTrainer


class RamaTrainer(DefaultTrainer):
    def _validate(self):
        self.model.eval()

        metric, loss = 0.0, 0.0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                prediction = self.model(x)
                loss += self.loss(prediction, y).item()
                this_metric, seg = self.metric(prediction, y)
                metric += this_metric.item()

        metric /= len(self.val_loader)
        loss /= len(self.val_loader)
        if self.logger is not None:
            self.logger.log_validation(self._iteration, metric, loss, x, y, prediction)
            self.logger.tb.add_image(tag="validation/rama-seg", img_tensor=seg[None], global_step=self._iteration)
        return metric
