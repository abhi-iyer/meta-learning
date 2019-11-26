from tools import *

class MAML():
    def __init__(self, output_dir, meta_iter=50, train_batch_size=20, test_batch_size=40):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.meta_iter = meta_iter
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        
        self.history = []
        self.train_loss = []
        self.train_return = []
        self.test_acc = []
        
        self.train_sampler = Sampler('RandomMiniEnv', meta_iter=meta_iter, 
                                     batch_size=train_batch_size, device=self.device)
        self.test_sampler = Sampler('RandomMiniEnv', meta_iter=meta_iter,
                                    batch_size=test_batch_size, device=self.device)
        
        self.policy = Policy()
        self.baseline = LinearFeatureBaseline(input_size=135)
        self.ml = MetaLearner(self.train_sampler, self.policy, 
                              self.baseline, num_episodes=train_batch_size, device=self.device)
        
        os.makedirs(output_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(output_dir, 
                                            "checkpoint.pth.tar")
        self.config_path = os.path.join(output_dir, "config.txt")
        
        locs = {k : v for k, v in locals().items() if k is not 'self'}
        self.__dict__.update(locs)
        
        if os.path.isfile(self.config_path):
            with open(self.config_path, 'r') as f:
                if f.read()[:-1] != repr(self):
                    print(f.read())
                    raise ValueError(
                        "Cannot create this experiment: "
                        "I found a checkpoint conflicting with the current setting.")
            self.load()
        else:
            self.save()
        
    @property
    def iteration(self):
        return len(self.history)
    
    def setting(self):
        return {'Policy' : self.policy,
                'Baseline' : self.baseline,
                'TrainBatchSize' : self.train_batch_size}  
    
    def __repr__(self):
        string = ''
        for key, val in self.setting().items():
            string += '{}({})\n'.format(key, val)
        return string
    
    def state_dict(self):
        return {'Policy' : self.policy.state_dict(),
                'Baseline' : self.baseline.state_dict(),
                'TrainSampler' : self.train_sampler,
                'History' : self.history,
                'TrainLoss' : self.train_loss,
                'TrainReturn' : self.train_return}
    
    def load_state_dict(self, checkpoint):
        self.policy.load_state_dict(checkpoint['Policy'])
        self.baseline.load_state_dict(checkpoint['Baseline'])
        
        self.train_sampler = checkpoint['TrainSampler']
        self.history = checkpoint['History']
        self.train_loss = checkpoint['TrainLoss']
        self.train_return = checkpoint['TrainReturn']
        
    def save(self):
        torch.save(self.state_dict(), self.checkpoint_path)
        with open(self.config_path, 'w') as f:
            print(self, file=f)
    
    def load(self):
        checkpoint = torch.load(self.checkpoint_path,
                                map_location=self.device)
        self.load_state_dict(checkpoint)
        del checkpoint  
        
    def run(self):
        start_iter = self.iteration
        
        print("Start/Continue training from iteration {}".format(start_iter))
        
        for i in range(start_iter, self.meta_iter):
            tasks = self.train_sampler.sample_tasks(low=1, 
                                                    high=20, 
                                                    num_tasks=10)
            episodes = self.ml.sample(tasks, first_order=True)
                        
            avg_return = self.ml.average_return(episodes)
            loss = self.ml.step(episodes)
            
            self.history.append(i)
            self.train_return.append(avg_return)
            self.train_loss.append(loss)
            
            self.save()
            
            print("Done with meta-iteration {}. Avg Return = {}, Loss = {}".format(i,
                                                                               avg_return,
                                                                               loss))
            
        print("Finished training for {} meta-iterations".format(self.meta_iter))
        
exp = MAML(output_dir="experiment1")

exp.run()
