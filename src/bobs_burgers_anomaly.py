"""
🍔 Is This Order Gonna Be a Problem?
Bob's Burgers Order Anomaly Detection

Maps to real-world: API abuse, cost anomalies, resource exhaustion, fraud detection
"""

import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Set style for beautiful plots
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class BobsBurgersAnomalyDetector:
    """
    Detects problematic orders using unsupervised learning.

    Blue-Team Mapping:
    - items_per_order → request_size / batch_size
    - time_since_last_order → request_frequency
    - order_time → time_of_day patterns
    - customer_frequency → user_activity_history
    - total_cost → resource_consumption
    - prep_time_estimate → processing_time
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.cluster_names = {
            0: "Teddy Regular 🔧",
            1: "Linda Chaos 🎵",
            2: "Late Night Hugo 🕐",
            -1: "Absolutely Not 🚨",
        }

    def generate_synthetic_data(self, n_samples=1000):
        """
        Generate Bob's Burgers order data.

        Real-world analogy: Historical API requests, job logs, transaction records
        """
        np.random.seed(42)

        data = []
        current_time = datetime(2024, 1, 1, 8, 0)

        # Customer profiles (like different user behavior patterns)
        customers = {
            "teddy": {"frequency": 0.95, "pattern": "regular"},
            "linda": {"frequency": 0.80, "pattern": "chaos"},
            "tina": {"frequency": 0.70, "pattern": "normal"},
            "gene": {"frequency": 0.65, "pattern": "normal"},
            "louise": {"frequency": 0.60, "pattern": "weird"},
            "hugo": {"frequency": 0.30, "pattern": "timing"},
            "random_customer": {"frequency": 0.10, "pattern": "normal"},
        }

        for i in range(n_samples):
            # Pick customer type (weighted)
            customer_weights = [0.15, 0.12, 0.15, 0.15, 0.10, 0.05, 0.28]
            customer = np.random.choice(list(customers.keys()), p=customer_weights)
            profile = customers[customer]

            # Generate order based on profile
            if profile["pattern"] == "regular":
                # Teddy: Same thing every day (normal baseline)
                items = np.random.randint(1, 3)
                cost = items * np.random.uniform(8, 12)
                prep_time = items * 5
                time_since_last = np.random.uniform(1400, 1500)  # ~daily

            elif profile["pattern"] == "chaos":
                # Linda: Chaotic but harmless
                items = np.random.randint(2, 8)
                cost = items * np.random.uniform(7, 15)
                prep_time = items * np.random.uniform(4, 8)
                time_since_last = np.random.uniform(180, 600)

            elif profile["pattern"] == "timing":
                # Hugo: Shows up at bad times
                items = np.random.randint(1, 4)
                cost = items * np.random.uniform(9, 13)
                prep_time = items * 6
                time_since_last = np.random.uniform(2000, 5000)
                # Force late hours
                current_time = current_time.replace(hour=np.random.choice([21, 22, 23]))

            elif profile["pattern"] == "weird":
                # Louise: Suspicious but not quite anomalous
                items = np.random.randint(3, 7)
                cost = items * np.random.uniform(6, 14)
                prep_time = items * np.random.uniform(5, 9)
                time_since_last = np.random.uniform(300, 900)

            else:
                # Normal customers
                items = np.random.randint(1, 5)
                cost = items * np.random.uniform(8, 12)
                prep_time = items * 5
                time_since_last = np.random.uniform(500, 3000)

            # Add some true anomalies (5% of data)
            if np.random.random() < 0.05:
                # "30 burgers in 2 minutes" scenarios
                anomaly_type = np.random.choice(
                    ["massive", "rapid", "cost_spike", "timing_weird"]
                )

                if anomaly_type == "massive":
                    items = np.random.randint(25, 50)
                    cost = items * np.random.uniform(8, 12)
                    prep_time = items * 5

                elif anomaly_type == "rapid":
                    items = np.random.randint(8, 15)
                    time_since_last = np.random.uniform(1, 30)  # Too fast
                    cost = items * np.random.uniform(8, 12)
                    prep_time = items * 5

                elif anomaly_type == "cost_spike":
                    items = np.random.randint(3, 8)
                    cost = items * np.random.uniform(50, 150)  # Way too expensive
                    prep_time = items * 5

                elif anomaly_type == "timing_weird":
                    items = np.random.randint(15, 30)
                    current_time = current_time.replace(
                        hour=np.random.choice([3, 4, 23])
                    )
                    cost = items * np.random.uniform(8, 12)
                    prep_time = items * 5

            # Simulate time progression
            current_time += timedelta(minutes=np.random.randint(5, 45))

            data.append(
                {
                    "order_id": f"ORD_{i:05d}",
                    "customer": customer,
                    "items_per_order": items,
                    "time_since_last_order": time_since_last,
                    "order_time": current_time.hour + current_time.minute / 60,
                    "customer_frequency": profile["frequency"],
                    "total_cost": cost,
                    "prep_time_estimate": prep_time,
                    "timestamp": current_time,
                }
            )

        df = pd.DataFrame(data)

        print("🍔 Generated Bob's Burgers Order Data")
        print(f"   Total orders: {len(df)}")
        print(f"   Features: {len(df.columns)}")
        print(f"   Expected anomalies: ~{int(n_samples * 0.05)}")

        return df

    def train(self, df):
        """
        Train the anomaly detector using DBSCAN (unsupervised clustering).

        Why DBSCAN?
        - No need to specify number of clusters
        - Automatically finds outliers (label = -1)
        - Works well for density-based anomaly detection
        - Perfect for "does this feel weird?" detection
        """
        # Select features for modeling
        feature_cols = [
            "items_per_order",
            "time_since_last_order",
            "order_time",
            "customer_frequency",
            "total_cost",
            "prep_time_estimate",
        ]

        X = df[feature_cols].values

        # Normalize features (crucial for distance-based methods)
        X_scaled = self.scaler.fit_transform(X)

        # Train DBSCAN
        # eps: maximum distance between samples to be considered neighbors
        # min_samples: minimum samples in neighborhood to form a cluster
        self.model = DBSCAN(eps=0.8, min_samples=10)
        df["cluster"] = self.model.fit_predict(X_scaled)

        # Map numeric clusters to Bob's sigh levels
        df["cluster_name"] = df["cluster"].map(
            lambda x: self.cluster_names.get(x, f"Cluster {x}")
        )

        # Stats
        anomaly_count = (df["cluster"] == -1).sum()
        anomaly_pct = (anomaly_count / len(df)) * 100

        print("\n📊 Training Complete")
        print(f"   Clusters found: {df['cluster'].nunique()}")
        print(f"   Anomalies detected: {anomaly_count} ({anomaly_pct:.1f}%)")
        print("\n   Cluster Distribution:")
        for cluster_id, count in df["cluster"].value_counts().sort_index().items():
            name = self.cluster_names.get(cluster_id, f"Cluster {cluster_id}")
            print(f"      {name}: {count} orders ({count/len(df)*100:.1f}%)")

        return df

    def analyze_anomalies(self, df):
        """
        Deep dive into what makes anomalies anomalous.
        This is what you'd present to your security/ops team.
        """
        anomalies = df[df["cluster"] == -1]
        normal = df[df["cluster"] != -1]

        print("\n🚨 Anomaly Analysis - 'Would Bob Sigh?' Report")
        print("=" * 60)

        metrics = {
            "items_per_order": "Order Size",
            "time_since_last_order": "Time Since Last (min)",
            "total_cost": "Total Cost ($)",
            "prep_time_estimate": "Prep Time (min)",
            "order_time": "Order Hour",
        }

        for col, label in metrics.items():
            normal_mean = normal[col].mean()
            anomaly_mean = anomalies[col].mean()
            diff_pct = ((anomaly_mean - normal_mean) / normal_mean) * 100

            print(f"\n{label}:")
            print(f"   Normal:    {normal_mean:.2f}")
            print(f"   Anomalies: {anomaly_mean:.2f}")
            print(f"   Diff:      {diff_pct:+.1f}%")

        # Show some example anomalies
        print("\n🔍 Sample 'Absolutely Not' Orders:")
        print("-" * 60)
        sample_anomalies = anomalies.sample(min(5, len(anomalies)))
        for idx, row in sample_anomalies.iterrows():
            print(f"\n   Order {row['order_id']}:")
            print(f"      Customer: {row['customer']}")
            print(f"      Items: {row['items_per_order']}")
            print(f"      Cost: ${row['total_cost']:.2f}")
            print(f"      Time: {row['order_time']:.1f} (hour)")
            print("      Bob's reaction: 😰")

    def visualize(self, df, output_path="../outputs/anomaly_analysis.png"):
        """
        Create beautiful visualizations.
        These would be your dashboards/alerts in production.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "🍔 Bob's Burgers: Is This Order Gonna Be a Problem?",
            fontsize=16,
            fontweight="bold",
        )

        # Color mapping
        colors = {
            -1: "#FF4444",  # Red for anomalies
            0: "#44AA44",  # Green for Teddy
            1: "#FFAA44",  # Orange for Linda
            2: "#4444FF",  # Blue for Hugo
        }

        # 1. Items vs Cost (classic scatter)
        ax = axes[0, 0]
        for cluster in df["cluster"].unique():
            mask = df["cluster"] == cluster
            label = self.cluster_names.get(cluster, f"Cluster {cluster}")
            ax.scatter(
                df[mask]["items_per_order"],
                df[mask]["total_cost"],
                c=colors.get(cluster, "#888888"),
                label=label,
                alpha=0.6,
                s=50,
            )
        ax.set_xlabel("Items per Order")
        ax.set_ylabel("Total Cost ($)")
        ax.set_title("Order Size vs Cost")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Time patterns
        ax = axes[0, 1]
        for cluster in df["cluster"].unique():
            mask = df["cluster"] == cluster
            ax.scatter(
                df[mask]["order_time"],
                df[mask]["items_per_order"],
                c=colors.get(cluster, "#888888"),
                alpha=0.6,
                s=50,
            )
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Items per Order")
        ax.set_title("Timing Patterns")
        ax.grid(True, alpha=0.3)

        # 3. Frequency patterns
        ax = axes[0, 2]
        for cluster in df["cluster"].unique():
            mask = df["cluster"] == cluster
            ax.scatter(
                df[mask]["time_since_last_order"],
                df[mask]["items_per_order"],
                c=colors.get(cluster, "#888888"),
                alpha=0.6,
                s=50,
            )
        ax.set_xlabel("Time Since Last Order (min)")
        ax.set_ylabel("Items per Order")
        ax.set_title("Request Frequency Patterns")
        ax.grid(True, alpha=0.3)

        # 4. Cost distribution
        ax = axes[1, 0]
        anomalies = df[df["cluster"] == -1]["total_cost"]
        normal = df[df["cluster"] != -1]["total_cost"]
        ax.hist(
            [normal, anomalies],
            bins=30,
            label=["Normal", "Anomalies"],
            color=["#44AA44", "#FF4444"],
            alpha=0.7,
        )
        ax.set_xlabel("Total Cost ($)")
        ax.set_ylabel("Frequency")
        ax.set_title("Cost Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. PCA visualization (2D projection of all features)
        ax = axes[1, 1]
        feature_cols = [
            "items_per_order",
            "time_since_last_order",
            "order_time",
            "customer_frequency",
            "total_cost",
            "prep_time_estimate",
        ]
        X = df[feature_cols].values
        X_scaled = self.scaler.transform(X)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        for cluster in df["cluster"].unique():
            mask = df["cluster"] == cluster
            label = self.cluster_names.get(cluster, f"Cluster {cluster}")
            ax.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                c=colors.get(cluster, "#888888"),
                label=label,
                alpha=0.6,
                s=50,
            )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
        ax.set_title("PCA: All Features Projection")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. Anomaly severity heatmap
        ax = axes[1, 2]
        anomalies = df[df["cluster"] == -1]
        if len(anomalies) > 0:
            severity_features = ["items_per_order", "total_cost", "prep_time_estimate"]
            severity_data = anomalies[severity_features].head(10)

            # Normalize for visualization
            severity_normalized = (severity_data - severity_data.min()) / (
                severity_data.max() - severity_data.min()
            )

            im = ax.imshow(severity_normalized.T, cmap="Reds", aspect="auto")
            ax.set_yticks(range(len(severity_features)))
            ax.set_yticklabels([f.replace("_", " ").title() for f in severity_features])
            ax.set_xlabel("Anomaly Sample")
            ax.set_title("Anomaly Feature Intensity")
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\n📈 Visualization saved to: {output_path}")

        return fig

    def production_scoring_example(self, df):
        """
        Show how this would work in production.
        Real-world: This would be your API endpoint or batch job.
        """
        print("\n🔧 Production Deployment Pattern")
        print("=" * 60)

        # Simulate new incoming order
        new_order = {
            "items_per_order": 35,
            "time_since_last_order": 15,
            "order_time": 22.5,
            "customer_frequency": 0.05,
            "total_cost": 380,
            "prep_time_estimate": 175,
        }

        print("\n📥 New Order Received:")
        print(f"   Items: {new_order['items_per_order']}")
        print(f"   Cost: ${new_order['total_cost']}")
        print(f"   Time: {new_order['order_time']:.1f} (10:30 PM)")
        print(f"   Last order: {new_order['time_since_last_order']} min ago")

        # Score it
        feature_cols = [
            "items_per_order",
            "time_since_last_order",
            "order_time",
            "customer_frequency",
            "total_cost",
            "prep_time_estimate",
        ]
        X_new = np.array([list(new_order.values())])
        X_new_scaled = self.scaler.transform(X_new)

        # Predict cluster
        cluster = self.model.fit_predict(
            np.vstack([self.scaler.transform(df[feature_cols].values), X_new_scaled])
        )[-1]

        cluster_name = self.cluster_names.get(cluster, f"Cluster {cluster}")

        print(f"\n🎯 Model Prediction: {cluster_name}")

        if cluster == -1:
            print("\n🚨 ALERT: Anomalous Order Detected!")
            print("   Recommended Action: Flag for review")
            print("   Bob's status: 😰 'Oh no...'")
            print("\n   Real-world mapping:")
            print("      → Trigger rate limiting")
            print("      → Send alert to on-call")
            print("      → Log for investigation")
            print("      → Potentially reject/throttle")
        else:
            print("\n✅ Normal Order Pattern")
            print("   Bob's status: 😊 'I got this'")

        # Feature importance for this prediction
        print("\n📊 Why this classification?")
        normal_stats = df[df["cluster"] != -1][feature_cols].mean()
        for feature, value in new_order.items():
            normal_val = normal_stats[feature]
            diff_pct = ((value - normal_val) / normal_val) * 100
            status = (
                "🔴" if abs(diff_pct) > 100 else "🟡" if abs(diff_pct) > 50 else "🟢"
            )
            print(
                f"   {status} {feature}: {value:.2f} (normal: {normal_val:.2f}, diff: {diff_pct:+.1f}%)"
            )


def main():
    """
    Run the complete Bob's Burgers anomaly detection pipeline.
    """
    print("=" * 70)
    print("🍔 BOB'S BURGERS ANOMALY DETECTION SYSTEM")
    print("   'Is This Order Gonna Be a Problem?'")
    print("=" * 70)

    # Initialize detector
    detector = BobsBurgersAnomalyDetector()

    # Generate synthetic data
    print("\n📦 Step 1: Generate Synthetic Order Data")
    print("-" * 70)
    df = detector.generate_synthetic_data(n_samples=1000)

    # Train model
    print("\n🤖 Step 2: Train Anomaly Detection Model")
    print("-" * 70)
    df = detector.train(df)

    # Analyze anomalies
    print("\n🔍 Step 3: Analyze Detected Anomalies")
    print("-" * 70)
    detector.analyze_anomalies(df)

    # Visualize
    print("\n📊 Step 4: Generate Visualizations")
    print("-" * 70)
    detector.visualize(df)

    # Production example
    print("\n🚀 Step 5: Production Scoring Demo")
    print("-" * 70)
    detector.production_scoring_example(df)

    # Save data for further analysis
    output_csv = "../data/bobs_burgers_orders.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n💾 Data saved to: {output_csv}")

    # Blue-team mapping summary
    print("\n" + "=" * 70)
    print("🛡️  BLUE-TEAM PRODUCTION MAPPING")
    print("=" * 70)
    print(
        """
This exact pattern applies to:

1. API Abuse Detection
   items_per_order → request_payload_size
   time_since_last → request_interval
   total_cost → compute_resources_used

2. Cost Anomaly Detection
   total_cost → cloud_spending
   prep_time → processing_duration
   customer_frequency → service_usage_pattern

3. Queue/Job Monitoring
   items_per_order → job_size
   prep_time → estimated_duration
   order_time → scheduled_time

4. Resource Exhaustion Precursors
   All features → early warning signals
   Anomaly detection → proactive alerting

The model asks: "Does this feel like one of THOSE requests?"
Not "Is this malicious?" — that's a different problem.

This is about: vibes, patterns, "hmm that's weird" detection.
"""
    )

    print("\n✅ Pipeline Complete!")
    print("   Check outputs/ and data/ for results")


if __name__ == "__main__":
    main()
