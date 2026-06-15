from generate import generate
prompt="Machine Learning is"
top_ks=[0,10,50]
with open(
    "outputs/topk_results.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(
        "Top_k Demo\n"
    )
    for k in top_ks:
        f.write(
            f"For top_k: {k}\n"
        )
        result=generate(
            prompt=prompt,
            top_k=k,
        )

        f.write(
            result
        )
        f.write("\n\n")
        f.write("-" * 60)
        f.write("\n\n")



