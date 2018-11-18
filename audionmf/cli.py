import os

import click as click

from audionmf.audio.audio_data import AudioData


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
        output_file = get_output_handle(filename, 'anmf')

    compress(input_file, output_file, filetype, 'anmf')

    input_file.close()
    output_file.close()


@cli.command(name='decompress')
@click.argument('input_file', type=click.File('rb'))
@click.argument('output_file', type=click.File('wb'), required=False)
@click.option('-t', '--filetype', type=click.Choice(['wav', 'flac']), default='wav')
def decompress_command(input_file, output_file, filetype):
    if output_file is None:
        output_file = get_output_handle(input_file.name, filetype)

    decompress(input_file, output_file, 'anmf', filetype)

    input_file.close()
    output_file.close()


@cli.command(name='debug')
@click.argument('input_file', type=click.File('rb'))
def debug_command(input_file):
    with open('debug.anmf', 'wb') as anmf_file:
        compress(input_file, anmf_file, 'wav', 'anmf')
    with open('debug.anmf', 'rb') as anmf_file, open('debug.wav', 'wb') as wav_file:
        decompress(anmf_file, wav_file, 'anmf', 'wav')
