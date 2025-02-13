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

def analyze_country_distribution(df, save_path=None):
    """Analyze and visualize the distribution of countries in the dataset"""
    
    # Calculate country distribution
    country_counts = df['country_code'].value_counts()
    total_startups = len(df)
    
    # Calculate percentages
    country_percentages = (country_counts / total_startups * 100).round(2)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot top 20 countries
    sns.barplot(x=country_percentages.head(20).values, 
                y=country_percentages.head(20).index, 
                ax=ax1,
                color=sns.color_palette()[0])
    
    ax1.set_title('Top 20 Countries by Number of Startups')
    ax1.set_xlabel('Percentage of Total Startups')
    ax1.set_ylabel('Country Code')
    
    # Add percentage labels
    for i, v in enumerate(country_percentages.head(20)):
        ax1.text(v, i, f' {v:.1f}%', va='center')
    
    # Create pie chart for broader distribution
    sizes = [
        country_percentages.head(1).sum(),  # Top 1
        country_percentages.iloc[1:5].sum(), # Next 4
        country_percentages.iloc[5:10].sum(), # Next 5
        country_percentages.iloc[10:].sum()  # Rest
    ]
    
    labels = [
        f'Top 1 Country ({country_percentages.index[0]})',
        'Next 4 Countries',
        'Next 5 Countries',
        f'Rest ({len(country_percentages)-10} countries)'
    ]
    
    colors = sns.color_palette("husl", 4)
    ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax2.set_title('Distribution of Startups Across Country Groups')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed statistics
    print("\nCountry Distribution Statistics:")
    print(f"Total number of countries: {len(country_counts)}")
    print(f"Number of countries with >1% of startups: {len(country_percentages[country_percentages > 1])}")
    
    print("\nTop 10 Countries:")
    for country, percentage in country_percentages.head(10).items():
        count = country_counts[country]
        print(f"{country}: {count:,} startups ({percentage:.1f}%)")
    
    # Calculate concentration metrics
    top_5_concentration = country_percentages.head(5).sum()
    print(f"\nConcentration Metrics:")
    print(f"Top 5 countries account for {top_5_concentration:.1f}% of all startups")
    
    # Calculate number of countries needed for different coverage levels
    cumsum = country_percentages.cumsum()
    print("\nCoverage Analysis:")
    for coverage in [50, 75, 90]:
        n_countries = len(cumsum[cumsum <= coverage]) + 1
        actual_coverage = cumsum.iloc[n_countries-1] if n_countries <= len(cumsum) else cumsum.iloc[-1]
        print(f"Need {n_countries} countries to reach {actual_coverage:.1f}% coverage")

if __name__ == "__main__":
    # Read data
    df = pd.read_csv('C:\\Users\\elote\\Repositories\\startup-success\\data\\interim\\cleaned_startups.csv')
    
    # Create figures directory if it doesn't exist
    import os
    os.makedirs('figures', exist_ok=True)
    
    # Set style
    set_style()
    
    # Generate visualization
    analyze_country_distribution(df, 'figures/country_distribution.png')