import os

import click as click


def get_output_handle(input_filename, filetype):
    raw_name = os.path.splitext(os.path.basename(input_filename))[0]
    target_name = raw_name + '.' + filetype
    return open(target_name, 'wb')


@click.group()
def cli():
    pass


@cli.command()
@click.argument('input_file', type=click.File('rb'))
@click.argument('output_file', type=click.File('wb'), required=False)
def compress(input_file, output_file):
    if output_file is None:
        output_file = get_output_handle(input_file.name, 'anmf')
    # TODO
    input_file.close()
    output_file.close()


@cli.command()
@click.argument('input_file', type=click.File('rb'))
@click.argument('output_file', type=click.File('wb'), required=False)
@click.option('-t', '--filetype', type=click.Choice(['wav', 'flac']), default='wav')
def decompress(input_file, output_file, filetype):
    if output_file is None:
        output_file = get_output_handle(input_file.name, filetype)
    # TODO
    input_file.close()
    output_file.close()
