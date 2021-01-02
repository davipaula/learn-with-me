import ffmpeg


def run():
    trim_file(20, 30)
    trim_file(60, 120)

    print("Finished")


def trim_file(start, end):
    file_path = "../../../learn_with_me/data/raw/audio/"
    output_path = "../../../learn_with_me/data/processed/audio/"

    input_name = "Reprogramming your brain to overcome fear - Olympia LePoint at TEDxPCC-1PV7Hy_8fhA.mp3"
    output_name = f"{start}-{end}-{input_name}"

    # "atrim" means "audio trim". This is needed because we're working with audio tracks only
    input_file = ffmpeg.input(file_path + input_name).filter(
        "atrim", start=start, end=end
    )

    output, _ = (
        input_file.output(output_path + output_name)
        .overwrite_output()
        .run(capture_stdout=True)
    )


if __name__ == "__main__":
    run()
