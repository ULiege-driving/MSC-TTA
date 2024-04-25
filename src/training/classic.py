from training.trainer import *


class ClassicNetworkTrainer(NetworkTrainer):

    def __init__(self, network, optimizer, scaler, criterion, device, args):
        super().__init__(network, optimizer, scaler, criterion, device, args)
        # self.optimizer = self.optimizer  # torch.optim.Adam(network.parameters(), lr=args.lr)

    def train_epoch(self, dataloader: DataLoader):
        if dataloader is None:
            return
        print('#'*10, ' Training over buffer for 1 epoch ', '#'*10)
        self.network.train()

        for i, (images, pseudos, masks, _, _) in enumerate(dataloader):
            torch.cuda.empty_cache()
            if self.args.supervision == 'teacher':
                targets = pseudos
            else:
                targets = masks
            with torch.cuda.amp.autocast():

                images = images.to(self.device)
                targets = targets.to(self.device)
                # print(images.shape, targets.shape)

                outputs = self.network.forward(images)



                loss = self.criterion(outputs, targets)
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            images = images.to("cpu")
            targets = targets.to("cpu")

