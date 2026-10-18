from src import generate_data, eda, train, evaluate, fairness, recommend

if __name__ == "__main__":
    generate_data.run()   # 1. build virtual company data
    eda.run()             # 2. EDA plots
    train.run()           # 3. train + save model
    evaluate.run()        # 4. metrics, confusion matrix, importance
    fairness.run()        # 5. bias audit
    recommend.run()       # 6. HR action plans
    print("\nDONE. Check images/ and outputs/ folders. Launch dashboard: streamlit run app.py")
