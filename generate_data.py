import os
import pandas as pd
import numpy as np

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# 1. Generate Movies Dataset (50 iconic style movies across 5 genres)
movies = [
    # --- SCI-FI ---
    {"id": 1, "title": "Interstellar Journey", "genre": "Sci-Fi", "description": "An astronaut travels through a wormhole in space to find a new home for humanity, exploring time dilation and distant planets."},
    {"id": 2, "title": "Cyber City 2099", "genre": "Sci-Fi", "description": "A neo-noir cyberpunk detective hunts down rogue artificial intelligence and replicants in a rain-slicked, neon-lit future metropolis."},
    {"id": 3, "title": "The Quantum Paradox", "genre": "Sci-Fi", "description": "A brilliant physicist accidentally triggers a time loop, forcing her to solve a reality-bending quantum mystery to save her timeline."},
    {"id": 4, "title": "Martian Horizon", "genre": "Sci-Fi", "description": "An astronaut is left stranded on a barren red planet and must use science, engineering, and sheer willpower to survive and contact Earth."},
    {"id": 5, "title": "Starlight Odyssey", "genre": "Sci-Fi", "description": "A spaceship crew encounters a mysterious alien artifact at the edge of the galaxy that challenges their perception of reality."},
    {"id": 6, "title": "Artificial Soul", "genre": "Sci-Fi", "description": "A tech CEO builds a highly advanced humanoid android, leading to a deep psychological battle over consciousness and human emotion."},
    {"id": 7, "title": "Dark Matter", "genre": "Sci-Fi", "description": "Astronauts explore a derelict spaceship only to discover a terrifying energy source that distorts space, gravity, and human minds."},
    {"id": 8, "title": "Galaxy Alliance", "genre": "Sci-Fi", "description": "A young pilot joins a rebel fleet to fight an oppressive galactic empire, engaging in epic space dogfights and interstellar battles."},
    {"id": 9, "title": "Parallel Skies", "genre": "Sci-Fi", "description": "A scientist discovers a portal to a parallel universe where earth is ruled by advanced, benevolent technological beings."},
    {"id": 10, "title": "The Machine Mind", "genre": "Sci-Fi", "description": "A supercomputer designed to govern a utopian city decides that humanity must be controlled for its own preservation."},

    # --- ACTION ---
    {"id": 11, "title": "Speed Limit", "genre": "Action", "description": "A high-octane action thriller about a rogue cop chasing bank robbers through city streets in fast cars and explosive encounters."},
    {"id": 12, "title": "Shadow Agent", "genre": "Action", "description": "An elite black-ops spy is betrayed by his agency and must go on the run, using close-quarters combat and stealth to uncover a conspiracy."},
    {"id": 13, "title": "The Last Gladiator", "genre": "Action", "description": "An epic historical action movie about a betrayed warrior fighting for survival and vengeance in brutal colosseum arena battles."},
    {"id": 14, "title": "Underworld War", "genre": "Action", "description": "A vigilante wages a one-man war against a powerful crime syndicate, featuring stylized gunfights and martial arts choreography."},
    {"id": 15, "title": "Apex Predator", "genre": "Action", "description": "A group of mercenaries on a rescue mission in a dense tropical jungle find themselves hunted by a mysterious, stealthy beast."},
    {"id": 16, "title": "Midnight Run", "genre": "Action", "description": "A mercenary is hired to transport a high-value target across a war-torn urban wasteland while evading heavily armed gangs."},
    {"id": 17, "title": "Iron Fist", "genre": "Action", "description": "A martial arts expert returns home to avenge his master's death, entering a deadly underground tournament ruled by corrupt bosses."},
    {"id": 18, "title": "Skyline Heist", "genre": "Action", "description": "A team of professional thieves plans a daring robbery of a high-security vault located on the top floor of a futuristic skyscraper."},
    {"id": 19, "title": "Rogue Command", "genre": "Action", "description": "A squad of elite soldiers is sent behind enemy lines to dismantle a nuclear launch facility, encountering heavy military resistance."},
    {"id": 20, "title": "Urban Justice", "genre": "Action", "description": "A retired detective returns to the streets to clean up his neighborhood after his former partner is murdered by street gangs."},

    # --- ROMANCE ---
    {"id": 21, "title": "Summer in Paris", "genre": "Romance", "description": "A charming romantic drama about an American artist and a French writer meeting by chance and falling in love over one summer in Paris."},
    {"id": 22, "title": "Letters to You", "genre": "Romance", "description": "A heartwarming love story centered on two pen pals who exchange letters for decades, sharing life's triumphs and heartbreaks before finally meeting."},
    {"id": 23, "title": "Waves of Love", "genre": "Romance", "description": "A small-town surfer falls in love with a visiting marine biologist, overcoming family expectations and contrasting lifestyles."},
    {"id": 24, "title": "Midnight Waltz", "genre": "Romance", "description": "An aspiring classical pianist and a passionate contemporary dancer find inspiration and romance in each other's artistic worlds."},
    {"id": 25, "title": "Autumn Whispers", "genre": "Romance", "description": "Two former childhood sweethearts reunite at a wedding in a picturesque New England town, rekindling old feelings and unresolved conflicts."},
    {"id": 26, "title": "Stargazing", "genre": "Romance", "description": "A quirky astrophysicist and a free-spirited travel writer bond over a shared road trip to witness a rare meteor shower."},
    {"id": 27, "title": "The Chef's Recipe", "genre": "Romance", "description": "A high-strung gourmet chef and an easy-going food critic clash professionally but spark a passionate romance in the kitchen."},
    {"id": 28, "title": "Chasing Sunsets", "genre": "Romance", "description": "A documentary photographer and a local tour guide explore remote tropical islands, discovering love amidst breathtaking scenery."},
    {"id": 29, "title": "City Lights & Hearts", "genre": "Romance", "description": "Two busy professionals running away from their pasts cross paths in a bustling city coffee shop and discover unexpected affection."},
    {"id": 30, "title": "Love Notes", "genre": "Romance", "description": "A music teacher helping a widowed songwriter finish his album finds herself falling in love as they write beautiful love ballads together."},

    # --- COMEDY ---
    {"id": 31, "title": "Grandma's Wild Ride", "genre": "Comedy", "description": "A hilarious comedy about a straight-laced accountant forced to take his eccentric, filter-free grandmother on a chaotic cross-country road trip."},
    {"id": 32, "title": "The Big Mix-Up", "genre": "Comedy", "description": "Two identical twins—one a high-profile corporate lawyer, the other a relaxed yoga instructor—swap lives for a week with disastrous, funny results."},
    {"id": 33, "title": "Superstore Chaos", "genre": "Comedy", "description": "An ensemble workplace comedy focusing on the quirky, colorful, and downright bizarre employees working the night shift at a 24-hour hypermarket."},
    {"id": 34, "title": "Accidental Heroes", "genre": "Comedy", "description": "Three clueless friends on a camping trip stumble into a smuggler's camp and are mistaken for elite federal agents, leading to slapstick survival."},
    {"id": 35, "title": "Chef Disasters", "genre": "Comedy", "description": "A comedy about a clumsy amateur home cook who enters a prestigious reality TV cooking competition and somehow advances through pure luck."},
    {"id": 36, "title": "The Office Prankster", "genre": "Comedy", "description": "An office worker orchestrates increasingly elaborate and funny pranks on his grumpy boss, culminating in a corporate retreat disaster."},
    {"id": 37, "title": "Virtual Reality Mishap", "genre": "Comedy", "description": "A gamer gets stuck in a comical, bug-ridden simulation game where they must defeat a ridiculous villain to escape back to the real world."},
    {"id": 38, "title": "Wedding Crashers' Club", "genre": "Comedy", "description": "A group of friends make a bet to see who can crash the most high-society weddings in a single summer, leading to hilarious awkward encounters."},
    {"id": 39, "title": "Stray Dog Shenanigans", "genre": "Comedy", "description": "A charming family comedy about a highly intelligent, mischievous stray dog who adopts a dysfunctional family and turns their lives upside down."},
    {"id": 40, "title": "Pet Detectives", "genre": "Comedy", "description": "Two goofy pet detectives are hired to find a billionaire's missing championship parrot, leading them into a wacky criminal underworld."},

    # --- THRILLER ---
    {"id": 41, "title": "Silent Witness", "genre": "Thriller", "description": "A deaf woman must outsmart a home intruder in a secluded forest cabin, relying on her wits and heightened senses to survive the night."},
    {"id": 42, "title": "Fractured Minds", "genre": "Thriller", "description": "A psychological thriller about a detective investigating a murder, who begins to suspect that his own mind is playing tricks on him."},
    {"id": 43, "title": "The Cabin", "genre": "Thriller", "description": "A group of friends spend the weekend at a remote cabin in the woods, only to find themselves hunted by a masked figure playing psychological games."},
    {"id": 44, "title": "Double Identity", "genre": "Thriller", "description": "A mild-mannered banker discovers he shares a name and bank account with a notorious international money launderer, putting a target on his back."},
    {"id": 45, "title": "Deep Water", "genre": "Thriller", "description": "A marine researcher stranded on a sinking research platform must evade a group of hostile mercenaries looking to steal her deep-sea data."},
    {"id": 46, "title": "The Perfect Crime", "genre": "Thriller", "description": "An meticulous art thief plans his final heist, but a relentless detective matches him step-for-step in a high-stakes psychological game of chess."},
    {"id": 47, "title": "Missing Link", "genre": "Thriller", "description": "A journalist investigating a corporate cover-up realizes that everyone she interviews is disappearing, making her the next target."},
    {"id": 48, "title": "Blind Spot", "genre": "Thriller", "description": "A blind programmer discovers a hidden malicious code in an autonomous car software, leading to a high-speed chase to stop a citywide disaster."},
    {"id": 49, "title": "The Safehouse", "genre": "Thriller", "description": "A government witness and a disgraced agent are trapped in a high-tech safehouse after its coordinates are leaked to assassin syndicates."},
    {"id": 50, "title": "Gridlock", "genre": "Thriller", "description": "A hostage negotiator is trapped in a massive traffic gridlock on a suspension bridge wired with explosives by an anonymous extortionist."}
]

df_movies = pd.DataFrame(movies)
df_movies.to_csv('data/movies.csv', index=False)
print(f"Saved {len(movies)} movies to data/movies.csv")

# 2. Generate Ratings Dataset (Collaborative Filtering)
# We will create 200 users. To make the recommendations meaningful, we will inject patterns:
# - Users 1-40: Sci-Fi Enthusiasts (rate Sci-Fi highly, Thrillers moderately, Romance poorly)
# - Users 41-80: Action & Thriller Fans (rate Action/Thirller highly, Comedy moderately, Romance/Sci-Fi poorly)
# - Users 81-120: Romance Lovers (rate Romance highly, Comedy moderately, Action/Sci-Fi poorly)
# - Users 121-160: Comedy Lovers (rate Comedy highly, Romance moderately, Thriller/Sci-Fi poorly)
# - Users 161-200: General/Random viewers (varied ratings)

np.random.seed(42)
ratings = []

for user_id in range(1, 201):
    # Determine user preference group
    if user_id <= 40:
        pref = "scifi"
    elif user_id <= 80:
        pref = "action_thriller"
    elif user_id <= 120:
        pref = "romance"
    elif user_id <= 160:
        pref = "comedy"
    else:
        pref = "random"

    # Number of movies rated by this user (between 8 and 20 movies)
    num_ratings = np.random.randint(8, 21)
    # Pick a random subset of movies
    selected_movie_ids = np.random.choice(df_movies['id'].values, size=num_ratings, replace=False)

    for movie_id in selected_movie_ids:
        movie_genre = df_movies[df_movies['id'] == movie_id]['genre'].values[0]
        
        # Base rating logic based on preference groups
        if pref == "scifi":
            if movie_genre == "Sci-Fi":
                rating = np.random.choice([4, 5], p=[0.3, 0.7])
            elif movie_genre == "Thriller":
                rating = np.random.choice([3, 4, 5], p=[0.4, 0.4, 0.2])
            elif movie_genre == "Romance":
                rating = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
            else:
                rating = np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2])
                
        elif pref == "action_thriller":
            if movie_genre in ["Action", "Thriller"]:
                rating = np.random.choice([4, 5], p=[0.3, 0.7])
            elif movie_genre == "Comedy":
                rating = np.random.choice([3, 4], p=[0.6, 0.4])
            elif movie_genre in ["Romance", "Sci-Fi"]:
                rating = np.random.choice([1, 2, 3], p=[0.5, 0.4, 0.1])
            else:
                rating = np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2])

        elif pref == "romance":
            if movie_genre == "Romance":
                rating = np.random.choice([4, 5], p=[0.2, 0.8])
            elif movie_genre == "Comedy":
                rating = np.random.choice([3, 4, 5], p=[0.3, 0.5, 0.2])
            elif movie_genre in ["Action", "Sci-Fi"]:
                rating = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
            else:
                rating = np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2])

        elif pref == "comedy":
            if movie_genre == "Comedy":
                rating = np.random.choice([4, 5], p=[0.2, 0.8])
            elif movie_genre == "Romance":
                rating = np.random.choice([3, 4, 5], p=[0.3, 0.5, 0.2])
            elif movie_genre in ["Thriller", "Sci-Fi"]:
                rating = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
            else:
                rating = np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2])

        else: # random
            rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.4, 0.2, 0.1])

        ratings.append({"user_id": user_id, "movie_id": int(movie_id), "rating": int(rating)})

df_ratings = pd.DataFrame(ratings)
df_ratings.to_csv('data/ratings.csv', index=False)
print(f"Saved {len(ratings)} user ratings to data/ratings.csv")
