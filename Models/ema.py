class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]


class EMA_teacher():
    def __init__(self, s_model, t_model, decay):
        self.s_model = s_model
        self.t_model = t_model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.s_model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.s_model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

        # for name, param in self.t_model.named_parameters():
        #     if param.requires_grad:
        #         assert name in self.shadow
        #         param.data = self.shadow[name]

    def apply_shadow(self):
        for name, param in self.t_model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                # self.backup[name] = param.data
                param.data = self.shadow[name]


    def update_parameters(self):
        pass


