from app.services.ad_service import AdService

def main():
    service = AdService()

    # Get user inputs to determine what type of Ad they want
    tone = input("Enter tone (fun, luxury, urgent, etc): ")
    audience = input("Enter target audience (students, commuters, tourists): ")
    platform = input("Enter platform (Instagram, Email, Google Ads): ")
    promotion = input("Enter promotion (discount, free trial, etc): ")
    bike_type = input("Enter bike type (electric, mountain, city): ")

    # Check for exit condition
    if (tone.lower() or audience.lower() or platform.lower() or promotion.lower() or bike_type.lower()) in ['exit', 'quit']:
        print('Exiting chatbot.')
        return 0

    print('Generating BikeEase Ads...')

    ads = service.generate_ads(
        tone=tone,
        audience=audience,
        platform=platform,
        bike_type=bike_type,
        promotion=promotion
    )

    print("\nGenerated Ads:\n")

    for i, ad in enumerate(ads, 1):
        print(f"--- Ad {i} ---\n{ad}\n")

if __name__ == "__main__":
    main()