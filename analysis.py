# Import necessary libraries
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class FinanceDataAnalyzer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = self.load_data()
        
    def load_data(self):
        path = kagglehub.dataset_download(self.dataset_path)
        return pd.read_csv(f"{path}/data.csv")

    def display_statistics(self):
        summary_stats = self.data.describe()
        correlation_matrix = self.data.corr(numeric_only=True)
        print("Summary Statistics:\n", summary_stats)
        print("\nCorrelation Matrix:\n", correlation_matrix)
        return summary_stats, correlation_matrix

    def plot_spending_distribution_pie(self):
        spending_columns = ['Groceries', 'Transport', 'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 'Education', 'Miscellaneous']
        total_spending = self.data[spending_columns].mean()
        plt.figure(figsize=(8, 8))
        plt.pie(total_spending, labels=total_spending.index, autopct='%1.1f%%', startangle=140)
        plt.title('Spending Distribution Across Categories')
        plt.tight_layout()
        plt.show()

    def plot_income_vs_disposable_income_with_line(self):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Income', y='Disposable_Income', data=self.data, alpha=0.7, color="green", label="Data Points")
        sns.regplot(x='Income', y='Disposable_Income', data=self.data, scatter=False, color="blue", line_kws={"label": "Trend Line"})
        plt.title('Income vs Disposable Income')
        plt.xlabel('Income')
        plt.ylabel('Disposable Income')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_loan_vs_city_tier(self):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='City_Tier', y='Loan_Repayment', data=self.data, palette="Set3")
        plt.title('Loan Repayment Across City Tiers')
        plt.xlabel('City Tier')
        plt.ylabel('Loan Repayment')
        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self):
        plt.figure(figsize=(20, 16))
        correlation_matrix = self.data.corr(numeric_only=True)
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Heatmap of Financial Variables')
        plt.tight_layout()
        plt.show()

# Initialize and use the FinanceDataAnalyzer class
analyzer = FinanceDataAnalyzer("shriyashjagtap/indian-personal-finance-and-spending-habits")
summary_stats, correlation_matrix = analyzer.display_statistics()
analyzer.plot_spending_distribution_pie()
analyzer.plot_income_vs_disposable_income_with_line()
analyzer.plot_loan_vs_city_tier()
analyzer.plot_correlation_heatmap()
