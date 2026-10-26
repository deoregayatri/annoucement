from ingestion.company_matcher import CompanyMatcher

print("Testing matcher...")

matcher = CompanyMatcher()

tests = [
    "Reliance Industries Ltd",
    "Infosys",
    "Nifty",
    "Random Company"
]

for t in tests:
    print("\nInput:", t)
    print("Output:", matcher.match(t))