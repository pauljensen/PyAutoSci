class FactorSet:
    def __init__(self):
        self.factors = []
            
    def add_continuous(self, name, minimum, maximum):
        self.factors.insert(0,[name, [minimum, maximum], "Continuous"])

    def add_ordinal(self, name, levels):
        self.factors.append([name, levels, "Ordinal"])

    def add_categorical(self, name, levels):
        self.factors.append([name, levels, "Categorical"])
