from generate import generate
prompt="Machine Learning is"
temperatures=[0.1,0.5,1,1.5]
with open(
    "outputs/temperature_results.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(
        "Temperature Demo\n"
    )
    for temp in temperatures:
        f.write(
            f"For temperature: {temp}\n"
        )
        result=generate(
            prompt=prompt,
            temperature=temp,
        )

        f.write(
            result
        )
        f.write("\n\n")
        f.write("-" * 60)
        f.write("\n\n")



