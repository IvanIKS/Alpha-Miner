import pandas as pd

class Requirement:
    def validate(self, strategy, bot):
        pass

class MustHaveSufficientAmountOfSymbols(Requirement):
    def validate(self, strategy, bot):
        if (len(bot.symbols) < strategy.required_symbols):
            raise Exception(f"Strategy requires at least {strategy.required_symbols} symbols, got only {len(bot.symbols)}")
        
class DataRequirement(Requirement):
    def validate(self, data):
        pass

class MustBePandasDataFrame(DataRequirement):
    def validate(self, data):
        if not isinstance(data, pd.DataFrame):
            raise Exception("Data must be a pandas DataFrame")

class MustHaveColumns(DataRequirement):
    def __init__(self, columns: set):
        self.required_columns = columns

    def validate(self, data):
        if not self.required_columns.issubset(data.columns):
            raise Exception(f"Data must contain the following columns: {self.required_columns}")

class MustBeBinanceOHLCVData(MustHaveColumns):
    def __init__(self):
        super().__init__({'open time', 'open', 'high', 'low', 'close', 'volume'})


        