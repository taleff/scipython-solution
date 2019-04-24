from scipython2 import luhn

#P4.6.1a
class BankAccount:
    
    def __init__(self, customer, account_number, balance = 0):
        self.customer = customer
        if luhn(account_number):
            self.account_number = account_number
        else:
            raise ValueError('Invalid account number: ', account_number)
        self.balance = balance
        
    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
        else:
            print('Invalid deposit amount:', amount)
        
    def withdraw(self, amount):
        if amount > 0:
            if amount > self.balance:
                print('Insufficient funds')
            else:
                self.balance -= amount
        else:
            print('Invalid withdrawal amount:', amount)