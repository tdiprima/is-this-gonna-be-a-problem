"""
Test your own orders through Bob's trained detector
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.bobs_burgers_anomaly import BobsBurgersAnomalyDetector


def test_custom_order():
    """
    Interactive tool to test custom orders
    """
    print("=" * 70)
    print("🍔 BOB'S BURGERS: CUSTOM ORDER TESTER")
    print("=" * 70)

    # Load trained model
    print("\n📦 Loading trained model...")
    detector = BobsBurgersAnomalyDetector()
    df = detector.generate_synthetic_data(n_samples=1000)
    df = detector.train(df)

    print("\n✅ Model loaded successfully!")
    print("\nNow let's test some interesting orders...\n")

    # Test cases
    test_cases = [
        {
            "name": "Bear's Regular Lunch",
            "order": {
                "items_per_order": 3,          # normal-sized order
                "time_since_last_order": 720,  # ~12 hours
                "order_time": 13.0,            # normal lunch
                "customer_frequency": 0.85,    # frequent regular
                "total_cost": 32.50,           # reasonable spend
                "prep_time_estimate": 18,      # standard prep
            },
            "expected": "🟢 Looks normal; no action needed",
        },
    ]

    # Score each test case
    feature_cols = [
        "items_per_order",
        "time_since_last_order",
        "order_time",
        "customer_frequency",
        "total_cost",
        "prep_time_estimate",
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST CASE {i}: {test['name']}")
        print("=" * 70)

        order = test["order"]

        # Display order details
        print("\n📋 Order Details:")
        print(f"   Items: {order['items_per_order']}")
        print(f"   Time since last: {order['time_since_last_order']} minutes")
        print(f"   Order time: {order['order_time']:.1f} (hour)")
        print(f"   Customer frequency: {order['customer_frequency']}")
        print(f"   Total cost: ${order['total_cost']:.2f}")
        print(f"   Prep time: {order['prep_time_estimate']} minutes")

        # Prepare features
        X_new = np.array([list(order.values())])
        X_new_scaled = detector.scaler.transform(X_new)

        # Predict
        all_features = detector.scaler.transform(df[feature_cols].values)
        X_combined = np.vstack([all_features, X_new_scaled])
        clusters = detector.model.fit_predict(X_combined)
        cluster = clusters[-1]

        cluster_name = detector.cluster_names.get(cluster, f"Cluster {cluster}")

        # Result
        print(f"\n🎯 Model Prediction: {cluster_name}")
        print(f"   Expected: {test['expected']}")

        if cluster == -1:
            print("\n🚨 BOB'S REACTION: 'Oh no... not again...'")
            print("   Status: ANOMALY DETECTED")
            print("   Recommendation: FLAG FOR REVIEW")
        else:
            print("\n😊 BOB'S REACTION: 'I can handle this'")
            print("   Status: Normal operation")

        # Feature deviation analysis
        print("\n📊 Feature Deviation from Normal:")
        normal_stats = df[df["cluster"] != -1][feature_cols].mean()

        for feature, value in order.items():
            normal_val = normal_stats[feature]
            diff_pct = ((value - normal_val) / normal_val) * 100

            if abs(diff_pct) > 100:
                status = "🔴 EXTREME"
            elif abs(diff_pct) > 50:
                status = "🟡 HIGH"
            else:
                status = "🟢 NORMAL"

            print(f"   {status} {feature}: {diff_pct:+.1f}% from baseline")


if __name__ == "__main__":
    test_custom_order()
    print("\n" + "=" * 70)
    print("✅ Testing Complete!")
    print("=" * 70)
    print("\nGo forth and detect weird vibes! 🛡️")
