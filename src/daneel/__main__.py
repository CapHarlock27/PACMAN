import datetime
import argparse
from daneel.parameters import Parameters

def main():
    parser = argparse.ArgumentParser()

    # Primary input argument
    parser.add_argument(
        "-i",
        "--input",
        dest = "input_files",
        type = str,
        nargs = "+",
        required = True,
        help = "Input file(s) to pass. For --transit, provide .yaml parameter files. For --dream, provide path to .pt PyTorch weights.",
    )

    # Modes
    parser.add_argument(
        "-t",
        "--transit",
        dest = "transit",
        required = False,
        help = "Plot the light curve of the selected exoplanet (requires .yaml parameter file(s) )",
        action = "store_true",
    )

    parser.add_argument(
        "--dream",
        dest = "dream",
        required = False,
        help = "'Dream' exoplanetary transit light curve(s) from a trained GAN (requires path to .pt weights)",
        action = "store_true",
    )
    
    parser.add_argument(
        "-d",
        "--detect",
        dest = "detect",
        type = str,                     
        required = False,
        help = "Initialise detection algorithms for Exoplanets (CNN or RF)",
    )

    parser.add_argument(
        "-a",
        "--atmosphere",
        dest = "atmosphere",
        type = str,
        required = False,
        help = "Atmospheric characterisazion from input transmission spectrum (model or retrieve)",
    )

    # Dedicated Dream arguments
    parser.add_argument(
        "--n_plots",
        dest = "n_plots",
        type = int,
        default = 1,
        help = "Number of samples to generate in Dream mode (default: 1)",
    )

    parser.add_argument(
        "-o",
        "--output",
        dest = "output_file",
        type = str,
        help = "Path and/or filename to save the generated plot(s)",
    )

    # Dedicated Atmosphere arguments
    parser.add_argument(
        "--plot",
        dest = "plot",
        type = bool,
        default = False,
        help = "Plot the atmospheric spectrum results (default: False)",
    )

    args = parser.parse_args()
    
    """Launch Daneel"""
    start = datetime.datetime.now()
    print(f"Daneel starts at {start}")

    if args.transit:
        from daneel.detection.transit_model import TransitModel
        if len(args.input_files) == 1:
            filename = args.input_files[0]
            input_params = Parameters(filename).params
            transit_section = input_params["transit"]

            model = TransitModel(transit_section)
            model.plot_light_curve()

        else:
            models = []
            for f in args.input_files:
                input_params = Parameters(f).params
                transit_section = input_params["transit"]
                models.append(TransitModel(transit_section))

            TransitModel.plot_multiple_light_curves(models)  
        transit = TransitModel(input_params['transit'])
        transit.plot_light_curve()

    elif args.detect == "rf":
        from daneel.detection.classifiers import RandomForestClassifier
        filename = args.input_files[0]
        input_params = Parameters(filename).params
        RF_class = RandomForestClassifier(input_params)
        RF_class.run()

    elif args.detect == "cnn":
        from daneel.detection.classifiers import CNNClassifier
        filename = args.input_files[0]
        input_params = Parameters(filename).params
        CNN_class = CNNClassifier(input_params)
        CNN_class.run()

    elif args.detect is not None:
        print(f"\nUnknown detection flag: {args.detect}")
        print("Valid options are: cnn or rf.")
    
    elif args.dream:
        from daneel.dream import Dream
        weights_path = args.input_files[0]
        n_plots = args.n_plots
        if args.output_file is None:
            output_file_name = "GAN_generated_transits.png"
        else:
            output_file_name = args.output_file
        print(f"Loading weights from: {weights_path}")
        print(f"Generating {n_plots} sample(s)...")
        dreamer = Dream(weights_path, n_plots)
        dreamer.dream(output_file = output_file_name)

    elif args.atmosphere == 'model':
        from daneel.atmosphere.forward_model import ForwardModel
        filename = args.input_files[0]
        plot = args.plot
        input_params = Parameters(filename).params
        if 'forward_model' in input_params:
            params = input_params['forward_model']
        else:
            params = input_params
        forward_model = ForwardModel(params)
        forward_model.run(plot)

    # elif args.atmosphere == 'retrieve':
    #     from daneel.atmosphere.retrieval import RetrievalModel

    elif args.atmosphere is not None:
        print(f"\nUnknown atmosphere flag: {args.atmosphere}")
        print("Valid options are: model or retrieve.")
    
    finish = datetime.datetime.now()
    print(f"Daneel finishes at {finish}")

if __name__ == "__main__":
    main()