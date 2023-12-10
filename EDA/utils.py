import regex as re
import numpy as np
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt


# Standarize to arff format
def arff_converter(input_file_path, output_file_path):
    with open(input_file_path, "r", encoding="utf-8") as input_file, open(
        output_file_path, "w", encoding="utf-8"
    ) as output_file:
        lines = input_file.readlines()
        for line in lines:
            splits = line.strip().split("\t")
            tag, sentence = splits
            sentence = sentence.replace(
                "'", ""
            )  # WEKA does not allow single quotes in sentences
            output_file.write(f"'{sentence}',{tag}\n")


# Create a test set with blind class
def test_blind_converter(input_file_path, output_file_path):
    with open(input_file_path, "r", encoding="utf-8") as input_file, open(
        output_file_path, "w", encoding="utf-8"
    ) as output_file:
        lines = input_file.readlines()
        for line in lines:
            line = line.replace(",ham", ",?")
            line = line.replace(",spam", ",?")
            output_file.write(line)


# Replace emoji symbols with [EMOJI] token
def emoji_token(input_file_path, output_file_path):
    with open(input_file_path, "r", encoding="utf-8") as input_file, open(
        output_file_path, "w", encoding="utf-8"
    ) as output_file:
        for line in input_file:
            patron = r"[:;][-]?[\']?[)(\*\$\|BDOSdPp/\\]+(?<![:/\\])"
            line = re.sub(patron, "[EMOJI]", line)
            output_file.write(line)


# Get instances for a sentence level analysis
def get_sentences(filepath):
    with open(filepath, "r", encoding="utf-8") as input_file:
        file_content = input_file.readlines()
        ham = []
        spam = []
        for line in file_content:
            if line.endswith(",ham\n"):
                parts = line.split(",ham\n")
                sentence = parts[0].replace("'", "")
                ham.append(sentence)
            elif line.endswith(",spam\n"):
                parts = line.split(",spam\n")
                sentence = parts[0].replace("'", "")
                spam.append(sentence)
    return ham, spam


# Get general statistic values for sentences
def get_stats(messages):
    messages = [(len(message)) for message in messages]
    print(f"Total sentences: {len(messages)}")
    print(f"Average sentence length: {round(np.mean(messages))}")
    print(f"Minimum sentence length: {min(messages)}")
    print(f"Maximum sentence length: {max(messages)}")
    print(f"Percentile 25, length: {np.percentile(messages, 25)}")
    print(f"Percentile 50, length: {np.percentile(messages, 75)}")


# Tokenize and prepare sentences for the analysis at a token level
def tokenizer(messages):
    tokens_list = []
    for message in messages:
        tokens_list.append(word_tokenize(message))
    tokens = []
    for sentence in tokens_list:
        for token in sentence:
            tokens.append(token)
    return tokens, tokens_list


# Prepare tokens for get_basic_tokens
def get_stats_tokens(messages):
    tokens, tokens_list = tokenizer(messages)
    return tokens, tokens_list


# Count tokens for each class
def get_basics_tokens(tokens_ham, tokens_list_ham, tokens_spam, tokens_list_spam):
    print(f"Total tokens: {len(tokens_ham) + len(tokens_spam)}")
    print(f"Spam tokens: {len(tokens_spam)}")
    print(f"Ham tokens: {len(tokens_ham)}")


# Get general statistic values for tokens
def get_advanced_tokens(tokens, tokens_list):
    avg_sentence_length = round(np.mean([len(sentence) for sentence in tokens_list]))
    min_sentence_length = min([len(sentence) for sentence in tokens_list])
    max_sentence_length = max([len(sentence) for sentence in tokens_list])
    print(f"Average sentence length (tokens): {avg_sentence_length}")
    print(f"Minimum sentence length (tokens): {min_sentence_length}")
    print(f"Maximum sentence length (tokens): {max_sentence_length}")
    percentile_25 = np.percentile([len(sentence) for sentence in tokens_list], 25)
    percentile_75 = np.percentile([len(sentence) for sentence in tokens_list], 75)
    print(f"Percentile 25, length: {percentile_25}")
    print(f"Percentile 75, length: {percentile_75}")


# Extract emojis before replacement
def extract_emojis(file_path):
    with open(file_path, "r", encoding="utf-8") as input_file:
        content = input_file.read()
        pattern = r"[:;][-]?[\']?[)(\*\$\|BDOSdPp/\\]+(?<![:/\\])"
        emojis_found = re.findall(pattern, content)
        return emojis_found


# Count emojis before replacement
def count_emojis(messages):
    total_emojis = 0
    for line in messages:
        pattern = r"[:;][-]?[\']?[)(\*\$\|BDOSdPp/\\]+(?<![:/\\])"
        emojis_found = re.findall(pattern, line)
        if emojis_found:
            total_emojis += len(emojis_found)
    return total_emojis


# Prepare sentences for the emoji analysis
def sentence_parser_emoji(file):
    ham_messages = []
    spam_messages = []
    with open(file, "r", encoding="utf-8") as input_file:
        lines = input_file.readlines()
        for line in lines:
            if line.endswith(",ham\n"):
                ham_messages.append(line)
            elif line.endswith(",spam\n"):
                spam_messages.append(line)
    return ham_messages, spam_messages


# Plot boxplot
def plot_custom_boxplots(ham_data, spam_data):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    box_colors = ["#87CEEB", "#FFA07A"]
    ham_box = axs[0].boxplot(
        ham_data,
        labels=["Ham"],
        showmeans=True,
        meanline=True,
        patch_artist=True,
        boxprops=dict(facecolor=box_colors[0], edgecolor="#2E5984", linewidth=1),
        whiskerprops=dict(color="#2E5984", linewidth=1),
        medianprops=dict(color="#2E5984", linewidth=1),
        capprops=dict(color="#2E5984", linewidth=1),
        meanprops=dict(color="#2E5984", linewidth=1),
    )
    axs[0].set_title("Ham SMSes", fontsize=16, fontweight="bold")
    axs[0].set_ylabel("Sentence Length (tokens)", fontsize=12)
    spam_box = axs[1].boxplot(
        spam_data,
        labels=["Spam"],
        showmeans=True,
        meanline=True,
        patch_artist=True,
        boxprops=dict(facecolor=box_colors[1], edgecolor="#A0522D", linewidth=1),
        whiskerprops=dict(color="#A0522D", linewidth=1),
        medianprops=dict(color="#A0522D", linewidth=1),
        capprops=dict(color="#A0522D", linewidth=1),
        meanprops=dict(color="#A0522D", linewidth=1),
    )
    axs[1].set_title("Spam SMSes", fontsize=16, fontweight="bold")
    axs[1].set_ylabel("Sentence Length (tokens)", fontsize=12)
    # Customize grid lines
    for ax in axs:
        ax.grid(True, linestyle="--", alpha=0.7)
        # Highlight outliers with red color
    for box in [ham_box]:
        for flier in box["fliers"]:
            flier.set(
                marker="o",
                markerfacecolor="#87CEEB",
                markeredgecolor="#2E5984",
                markersize=8,
            )
    for box in [spam_box]:
        for flier in box["fliers"]:
            flier.set(
                marker="o",
                markerfacecolor="#FFA07A",
                markeredgecolor="#A0522D",
                markersize=8,
            )
        # Increase font size for legend
    legend_labels = ["Median", "Mean"]
    axs[0].legend(
        [ham_box["medians"][0], ham_box["means"][0]], legend_labels, fontsize=10
    )
    axs[1].legend(
        [spam_box["medians"][0], spam_box["means"][0]], legend_labels, fontsize=10
    )
    # Set y-axis ticks for the second subplot based on the first subplot
    # axs[1].set_yticks(axs[0].get_yticks())

    plt.tight_layout()
    plt.show()
