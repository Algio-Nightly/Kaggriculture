import time
from kaggle_environments import make
from main import agent

def run_and_render():
    total_turns = 96  # 4 days
    print(f"Running simulation with agent vs starter for {total_turns} turns...")
    env = make("kaggriculture", configuration={"episodeSteps": total_turns}, debug=True)
    env.run([agent, "starter"])

    # Render HTML visualizer
    html_content = env.render(mode="html", width=1000, height=800)
    
    with open("replay.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print("Generated replay.html successfully!")

if __name__ == "__main__":
    run_and_render()
