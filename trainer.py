import torch

class Trainer():
    def initialize(
            self,
            train_loader,
            valid_loader,
            test_loader,
            encoder,
            classifier,
            optimizer,
            lr_scheduler,
            loss_function,
            accuracy_function,
            method
            ):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.encoder = encoder
        self.classifier = classifier
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function
        self.accuracy_function = accuracy_function
        self.method = method
        
        if method not in ['linear', 'finetune']:
            raise TypeError(f"method {self.method}")

    def one_epoch_train(self):
        loss_list = list()
        acc_list = list()

        if self.method == 'linear':
            # Set BatchNorm, Dropout to eval mode
            self.encoder.eval()
        elif self.method == 'finetune':
            self.encoder.train()
            
            
        self.classifier.train()
        
        total_size = 0
        for images, labels in self.train_loader:
            images = images.cuda()
            labels = labels.cuda()
            batch_size = labels.shape[0]
            total_size += batch_size

            features = self.encoder(images)
            outputs = self.classifier(features)
            loss = self.loss_function(outputs, labels)

            # Update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_list.append(loss.detach().cpu().item()*batch_size)
            accuracy = self.accuracy_function(outputs.detach().cpu(), labels.cpu())
            acc_list.append(accuracy.item()*batch_size)
        #self.lr_scheduler.step()
        return {
            'train loss' : sum(loss_list) / total_size,
            'train acc' : sum(acc_list) / total_size,
        }

    def one_epoch_valid(self):
        # Freeze parameters
        with torch.no_grad():
            loss_list = list()
            acc_list = list()
            # Freezing BatchNorm
            self.encoder.eval()
            self.classifier.eval()
            total_size = 0

            for images, labels in self.valid_loader:
                images = images.cuda()
                labels = labels.cuda()
                batch_size = labels.shape[0]
                total_size += batch_size

                
                features = self.encoder(images)
                outputs = self.classifier(features)
                loss = self.loss_function(outputs, labels)

                loss_list.append(loss.detach().cpu().item()*batch_size)
                accuracy = self.accuracy_function(outputs.detach().cpu(), labels.cpu())
                acc_list.append(accuracy.item()*batch_size)
                

            return {
                'valid loss' : sum(loss_list) / total_size,
                'valid acc' : sum(acc_list) / total_size,
            }

    def test(self):
        self.valid_loader = self.test_loader
        result_dict = self.one_epoch_valid()
        
        return {
            'test loss' : result_dict['valid loss'],
            'test acc' : result_dict['valid acc']
        }
    

    def save(self, path, others = None):
        state = {
            'encoder' : self.encoder.state_dict(),
            'classifier' : self.classifier.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'others' : others
        }
        torch.save(state, path)
        del state
    
    def load(self, path):
        state = torch.load(path)
        self.encoder.load_state_dict(state['encoder'])
        self.classifier.load_state_dict(state['classifier'])
        del state
        
    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']