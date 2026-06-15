from generate import generate
prompt="Machine Learning is"
top_ps=[0.1,0.5,0.95]
with open(
    "outputs/topp_results.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(
        "Top_p Demo\n"
    )
    for p in top_ps:
        f.write(
            f"For top_p: {p}\n"
        )
        result=generate(
            prompt=prompt,
            top_p=p,
        )

        f.write(
            result
        )
        f.write("\n\n")
        f.write("-" * 60)
        f.write("\n\n")



