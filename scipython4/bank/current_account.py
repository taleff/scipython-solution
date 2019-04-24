import bank_account

#P4.6.1b
class CurrentAccount(BankAccount):
    
    def __init__(self, customer, account_number, annual_fee, 
                 transaction_limit, overdraft_limit, balance = 0):
        self.annual_fee = annual_fee
        self.transaction_limit = transaction_limit
        self.overdraft_limit = overdraft_limit
        super().__init__(customer, account_number, balance)
        
    def withdraw(self, amount):
        
        if amount <= 0:
            print('Invalid withdrawal amount:', amount)
            return
            
        if amount > self.balance + self.overdraft_limit:
            print('Insifficient funds')
            return
        
        if amount > self.transation_limit:
            print('{0:s}{1:.2f} exceeds the single transaction limit of'
                  ' {0:s}{2:.2f}'.format(self.currency, amount, 
                  self.transation_limit))
            return
            
        self.balance -= amount
        
        def apply_annual_fee(self):
            self.balance = max(0., self.balance = self.annual_fee)