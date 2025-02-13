import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def set_style():
    """Set consistent style for all plots"""
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12

def plot_status_distribution(df, save_path=None):
    """Plot distribution of startup statuses"""
    plt.figure(figsize=(10, 6))
    
    # Calculate percentages
    status_counts = df['status'].value_counts()
    status_percentages = (status_counts / len(df) * 100).round(2)
    
    # Create bar plot
    ax = sns.barplot(x=status_percentages.index, y=status_percentages.values)
    
    # Add percentage labels on bars
    for i, v in enumerate(status_percentages.values):
        ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.title('Distribution of Startup Statuses')
    plt.xlabel('Status')
    plt.ylabel('Percentage of Startups')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print exact counts and percentages
    print("\nStartup Status Distribution:")
    for status, count in status_counts.items():
        percentage = status_percentages[status]
        print(f"{status}: {count:,} startups ({percentage:.1f}%)")

def plot_status_by_region(df, save_path=None):
    """Plot status distribution across regions"""
    plt.figure(figsize=(15, 8))
    
    # Get top 20 regions by number of startups
    top_regions = df['region'].value_counts().head(20).index
    df_top = df[df['region'].isin(top_regions)]
    
    # Calculate percentages within each region
    status_by_region = pd.crosstab(df_top['region'], df_top['status'], normalize='index') * 100
    
    # Plot stacked bars
    status_by_region.plot(kind='bar', stacked=True)
    
    plt.title('Startup Status Distribution by Top 20 Regions')
    plt.xlabel('Region')
    plt.ylabel('Percentage')
    plt.legend(title='Status', bbox_to_anchor=(1.05, 1))
    plt.xticks(rotation=45, ha='right')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print regions with significant imbalances
    print("\nStatus Distribution in Top 20 Regions:")
    print(status_by_region.round(2))

def plot_status_by_country(df, top_n=10, save_path=None):
    """Plot status distribution for top N countries by number of startups"""
    # Get top N countries by startup count
    top_countries = df['country_code'].value_counts().head(top_n).index
    df_top = df[df['country_code'].isin(top_countries)]
    
    plt.figure(figsize=(15, 8))
    
    # Calculate percentages within each country
    status_by_country = pd.crosstab(df_top['country_code'], df_top['status'], 
                                   normalize='index') * 100
    
    # Plot stacked bars
    status_by_country.plot(kind='bar', stacked=True)
    
    plt.title(f'Startup Status Distribution in Top {top_n} Countries')
    plt.xlabel('Country')
    plt.ylabel('Percentage')
    plt.legend(title='Status', bbox_to_anchor=(1.05, 1))
    plt.xticks(rotation=45, ha='right')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print countries with extreme distributions
    print(f"\nStatus Distribution in Top {top_n} Countries:")
    print(status_by_country.round(2))

def plot_category_balance(df, save_path=None):
    """Plot distribution of startup categories"""
    # Split the category_list string and explode to get individual categories
    categories = df['category_list'].str.split('|').explode()
    
    plt.figure(figsize=(15, 8))
    
    # Get top 20 categories
    top_cats = categories.value_counts().head(20)
    
    # Create bar plot
    ax = sns.barplot(x=top_cats.values, y=top_cats.index)
    
    # Add count labels
    for i, v in enumerate(top_cats.values):
        ax.text(v, i, f' {v:,}', va='center')
    
    plt.title('Top 20 Startup Categories')
    plt.xlabel('Number of Startups')
    plt.ylabel('Category')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print category distribution stats
    print("\nCategory Distribution Statistics:")
    total_startups = len(df)
    print(f"Total unique categories: {len(categories.unique())}")
    print(f"Average categories per startup: {len(categories)/total_startups:.2f}")

def main():
    # Read data
    df = pd.read_csv('C:\\Users\\elote\\Repositories\\startup-success\\data\\interim\\cleaned_startups.csv')
    
    # Create figures directory if it doesn't exist
    import os
    os.makedirs('figures', exist_ok=True)
    
    # Set style
    set_style()
    
    # Generate plots
    print("\nGenerating visualizations for class imbalance analysis...")
    
    # 1. Overall status distribution
    print("\n1. Analyzing overall status distribution...")
    plot_status_distribution(df, 'figures/status_distribution.png')
    
    # 2. Regional distribution
    print("\n2. Analyzing status distribution by region...")
    plot_status_by_region(df, 'figures/status_by_region.png')
    
    # 3. Country distribution
    print("\n3. Analyzing status distribution by country...")
    plot_status_by_country(df, top_n=10, save_path='figures/status_by_country.png')
    
    # 4. Category distribution
    print("\n4. Analyzing category distribution...")
    plot_category_balance(df, 'figures/category_distribution.png')

if __name__ == "__main__":
    main()