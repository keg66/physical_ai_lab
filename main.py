from physical_ai.agent import Agent

agent = Agent("output/scene_1.png")

result = agent.run("一番左の赤い箱を青い箱の隣に置いて")

print()
print("==== Result ====")
print(result)
