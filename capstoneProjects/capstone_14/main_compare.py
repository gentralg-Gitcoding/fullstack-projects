from app.services.lc_ad_service import AdService
from app.services.compare_service import EvalService
from app.services.hf_ad_service import HFAdService

def get_best(results):
    return max(results, key=lambda x: x[1]["total"])

def main():
    ad_service = AdService()          # LangChain model
    hf_ad_service = HFAdService()       # HF model
    eval_service = EvalService()

    tone="fun",
    audience="students",
    platform="instagram",
    bike_type="city",
    promotion="free rides"

    ads_langchain = ad_service.generate_ads(
        tone=tone,
        audience=audience,
        platform=platform,
        bike_type=bike_type,
        promotion=promotion
    )

    ads_hf = hf_ad_service.generate_ads(
        tone=tone,
        audience=audience,
        platform=platform,
        bike_type=bike_type,
        promotion=promotion
    )

    print("\nScoring LangChain Ads...\n")
    scored_lc = eval_service.score_ads(ads_langchain)

    print("\nScoring HF Ads...\n")
    scored_hf = eval_service.score_ads(ads_hf)

    best_lc = get_best(scored_lc)
    best_hf = get_best(scored_hf)

    print("\n=== RESULTS ===\n")

    print("Best LangChain Ad:\n", best_lc[0])
    print("Score:", best_lc[1])

    print("\nBest HF Ad:\n", best_hf[0])
    print("Score:", best_hf[1])

    if best_lc[1]["total"] > best_hf[1]["total"]:
        print("\nLangChain Model Wins!")
    else:
        print("\nHuggingFace Script Wins!")

if __name__ == "__main__":
    main()