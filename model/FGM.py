class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
            self.backup = {}

fgm = FGM(model)
 for batch_input, batch_label in data:
       loss = model(batch_input, batch_label)
       loss.backward()

       # adversarial training
       fgm.attack()
       loss_adv = model(batch_input, batch_label)
       loss_adv.backward()
       fgm.restore()

       optimizer.step()
       model.zero_grad()