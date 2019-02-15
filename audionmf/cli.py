import os

import click as click

from audionmf.audio.audio_data import AudioData, compression_schemes
from audionmf.util.plot_util import plot_signal


def get_filename_ext(path):
    return os.path.splitext(os.path.basename(path))


def get_output_handle(input_filename, filetype):
    raw_name = get_filename_ext(input_filename)[0]
    target_name = raw_name + '.' + filetype
    return open(target_name, 'wb')


def compress(input_file, output_file, audio_filetype, compression_filetype):
    audio = AudioData.from_audio_file(input_file, audio_filetype)

    if audio is None:
        print('invalid file format: {}'.format(audio_filetype))
    else:
        audio.write_compressed_file(output_file, compression_filetype)


def decompress(input_file, output_file, compression_filetype, audio_filetype):
    audio = AudioData.from_compressed_file(input_file, compression_filetype)
    audio.write_audio_file(output_file, audio_filetype)


@click.group()
def cli():
    pass


@cli.command(name='compress')
@click.argument('input_file', type=click.File('rb'))
@click.argument('output_file', type=click.File('wb'), required=False)
def compress_command(input_file, output_file):
    filename = input_file.name
    filetype = get_filename_ext(filename)[1].lower()[1:]
    if output_file is None:
        output_file = get_output_handle(filename, 'anmfs')

    compress(input_file, output_file, filetype, 'anmfs')  # TODO get extension automatically

    input_file.close()
    output_file.close()


@cli.command(name='decompress')
@click.argument('input_file', type=click.File('rb'))
@click.argument('output_file', type=click.File('wb'), required=False)
@click.option('-t', '--filetype', type=click.Choice(['wav', 'flac']), default='wav')
def decompress_command(input_file, output_file, filetype):
    if output_file is None:
        output_file = get_output_handle(input_file.name, filetype)

    decompress(input_file, output_file, 'anmfs', filetype)  # TODO get extension automatically

    input_file.close()
    output_file.close()


@cli.command(name='debug')
def debug_command():
    debug_path = 'debug'
    example_path = 'examples'

    if not os.path.exists(debug_path):
        os.makedirs(debug_path)

    sample_filenames = list()
    for file in os.listdir(example_path):
        if file.endswith('.wav'):
            sample_filenames.append(os.path.splitext(file)[0])

    for filename in sample_filenames:
        print('Processing {}...'.format(filename))

        with open(os.path.join(example_path, filename + '.wav'), 'rb') as input_file:
            audio = AudioData.from_audio_file(input_file, 'wav')

        sample_signal_in = audio.channels[0].samples
        plot_signal(sample_signal_in, os.path.join(debug_path, '{}_sig_in.png'.format(filename)))

        schemes = compression_schemes.keys()

        # debug
        # schemes = ['anmfs']

        for scheme in schemes:
            comp_path = '{}_com.{}'.format(filename, scheme)

            with open(os.path.join(debug_path, comp_path), 'wb') as comp_file:
                audio.write_compressed_file(comp_file, scheme)

            with open(os.path.join(debug_path, comp_path), 'rb') as comp_file:
                comp_audio = AudioData.from_compressed_file(comp_file, scheme)

            sample_signal_out = comp_audio.channels[0].samples
            plot_signal(sample_signal_out, os.path.join(debug_path, '{}_sig_{}_out.png'.format(filename, scheme)))

            with open(os.path.join(debug_path, '{}_dec_{}.wav'.format(filename, scheme)), 'wb') as output_file:
                comp_audio.write_audio_file(output_file, 'wav')
