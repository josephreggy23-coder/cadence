function export_h1r_astrocytes(raw_dir, output_dir)
%EXPORT_H1R_ASTROCYTES Convert Dryad MATLAB tables to a compact long table.
%
% The source .mat files use MATLAB's v7.3 table serialization.  This exporter
% deliberately uses MATLAB itself rather than depending on undocumented HDF5
% object internals.  It emits:
%
%   data/processed/h1r_astrocytes_v1.csv.gz
%   data/processed/h1r_astrocytes_v1_provenance.json
%
% Each output row is one fluorescence sample from one ROI.  ROI, slice,
% genotype, stimulus, protocol cohort, onset, and Dryad provenance remain
% explicit, so downstream code can respect the ROI -> slice hierarchy.
%
% Usage from the repository root:
%   matlab -batch "addpath('scripts'); export_h1r_astrocytes"

if nargin < 1
    raw_dir = fullfile('data', 'real');
end
if nargin < 2
    output_dir = fullfile('data', 'processed');
end

sampling_rate_hz = 0.71;
schema_version = 'h1r-astrocytes-v1';
dataset_doi = '10.5061/dryad.2280gb64x';
dryad_version_id = uint32(394734);

sources = struct( ...
    'file_name', { ...
        'Fig5_H1RKO_NE.mat', ...
        'Fig5_H1RKO_NE_postLowHA.mat'}, ...
    'file_id', {uint32(4342955), uint32(4342954)}, ...
    'sha256', { ...
        '45c7836bcafda6369a8addf436c2838c04ab61c53d3881fe52416821fe5e8116', ...
        'd39c8e8453996d5f3abdc42eed0b4ea4b6485b5e142719d4128f9683972621eb'}, ...
    'cohort', { ...
        'ne_only_2023', ...
        'post_low_histamine_2025'});

if ~isfolder(output_dir)
    mkdir(output_dir);
end

rows = table();
for source_index = 1:numel(sources)
    source = sources(source_index);
    input_path = fullfile(raw_dir, source.file_name);
    if ~isfile(input_path)
        error('Missing source file: %s', input_path);
    end

    actual_sha256 = sha256_file(input_path);
    if ~strcmpi(actual_sha256, source.sha256)
        error('SHA-256 mismatch for %s. Expected %s, found %s.', ...
            input_path, source.sha256, actual_sha256);
    end

    loaded = load(input_path);
    top_level_names = fieldnames(loaded);
    payload = loaded.(top_level_names{1});

    if isfield(payload, 'NE_Onsets')
        onset_table = payload.NE_Onsets;
    else
        onset_table = payload.Onsets;
    end
    onset_names = string(onset_table{:, 1});
    onset_values = double(onset_table{:, 2});

    table_names = fieldnames(payload.Dff_allSlices);
    for table_index = 1:numel(table_names)
        source_table = table_names{table_index};
        trace_table = payload.Dff_allSlices.(source_table);

        if strcmp(source.cohort, 'ne_only_2023')
            % The legacy struct names were truncated at MATLAB's identifier
            % limit, so match their unique prefix to the complete onset name.
            onset_row = find(startsWith(onset_names, string(source_table)), 1);
            stimulus = "NE";
            slice_id = erase(onset_names(onset_row), "_NE");
        else
            unsuffixed_name = regexprep(source_table, '_csv$', '');
            onset_row = find(onset_names == string(unsuffixed_name), 1);
            stimulus_match = regexp(unsuffixed_name, '_(HA|NE)$', 'tokens', 'once');
            stimulus = string(stimulus_match{1});
            slice_id = string(regexprep(unsuffixed_name, '_(HA|NE)$', ''));
        end
        if isempty(onset_row)
            error('No onset metadata matched table %s', source_table);
        end
        onset_index = onset_values(onset_row);

        variable_names = trace_table.Properties.VariableNames;
        mean_names = variable_names(startsWith(variable_names, 'Mean_'));
        for roi_index = 1:numel(mean_names)
            mean_name = mean_names{roi_index};
            roi_token = regexprep(mean_name, '^Mean_|_$', '');
            genotype = "KO";
            if contains(roi_token, 'WT')
                genotype = "WT";
            end
            roi_number_match = regexp(roi_token, '\d+$', 'match', 'once');
            if isempty(roi_number_match)
                error('Could not parse ROI number from %s', mean_name);
            end
            roi_id = genotype + string(roi_number_match);

            area_name = strrep(mean_name, 'Mean_', 'Area_');
            fluorescence = double(trace_table.(mean_name));
            area_values = double(trace_table.(area_name));
            area_pixels = median(area_values, 'omitnan');
            source_frames = double(trace_table{:, 1});
            sample_count = height(trace_table);
            frame_index = (0:(sample_count - 1))';
            time_from_onset_s = (frame_index - onset_index) ./ sampling_rate_hz;

            block = table( ...
                repmat(string(schema_version), sample_count, 1), ...
                repmat(string(dataset_doi), sample_count, 1), ...
                repmat(dryad_version_id, sample_count, 1), ...
                repmat(source.file_id, sample_count, 1), ...
                repmat(string(source.file_name), sample_count, 1), ...
                repmat(string(source.sha256), sample_count, 1), ...
                repmat(string(source.cohort), sample_count, 1), ...
                repmat(string(source_table), sample_count, 1), ...
                repmat(slice_id, sample_count, 1), ...
                repmat(stimulus, sample_count, 1), ...
                repmat(genotype, sample_count, 1), ...
                repmat(roi_id, sample_count, 1), ...
                repmat(string(mean_name), sample_count, 1), ...
                repmat(area_pixels, sample_count, 1), ...
                repmat(onset_index, sample_count, 1), ...
                repmat(sampling_rate_hz, sample_count, 1), ...
                frame_index, source_frames, time_from_onset_s, fluorescence, ...
                'VariableNames', { ...
                    'schema_version', 'source_doi', 'dryad_version_id', ...
                    'source_file_id', 'source_file', 'source_sha256', ...
                    'cohort', 'source_table', 'slice_id', 'stimulus', ...
                    'genotype', 'roi_id', 'source_roi_column', 'area_pixels', ...
                    'onset_index', 'sampling_rate_hz', 'frame_index', ...
                    'source_frame', 'time_from_onset_s', 'raw_fluorescence'});
            rows = [rows; block]; %#ok<AGROW>
        end
    end
end

csv_path = fullfile(output_dir, 'h1r_astrocytes_v1.csv');
gz_path = strcat(csv_path, '.gz');
if isfile(gz_path)
    delete(gz_path);
end
writetable(rows, csv_path);
script_dir = fileparts(mfilename('fullpath'));
compressor = fullfile(script_dir, 'deterministic_gzip.py');
command = sprintf('python "%s" "%s" "%s"', compressor, csv_path, gz_path);
[compression_status, compression_output] = system(command);
if compression_status ~= 0
    error('Deterministic gzip failed: %s', compression_output);
end
delete(csv_path);

manifest = struct();
manifest.schema_version = schema_version;
manifest.dataset_doi = dataset_doi;
manifest.dryad_version_id = dryad_version_id;
manifest.sampling_rate_hz = sampling_rate_hz;
manifest.derived_file = 'h1r_astrocytes_v1.csv.gz';
manifest.row_count = height(rows);
manifest.column_count = width(rows);
manifest.onset_convention = [ ...
    'onset_index is retained as the zero-based array index used by the ', ...
    'authors'' accompanying Python notebook; source_frame preserves the ', ...
    'first MATLAB table column separately.'];
manifest.hierarchy = 'samples nested in ROIs; ROIs nested in slices; animal IDs unavailable';
manifest.sources = sources;

manifest_path = fullfile(output_dir, 'h1r_astrocytes_v1_provenance.json');
file_id = fopen(manifest_path, 'w');
if file_id < 0
    error('Could not create %s', manifest_path);
end
cleanup = onCleanup(@() fclose(file_id));
fwrite(file_id, jsonencode(manifest, PrettyPrint=true), 'char');
clear cleanup;

fprintf('Exported %d samples to %s\n', height(rows), gz_path);
fprintf('Wrote provenance to %s\n', manifest_path);
end

function digest = sha256_file(file_path)
%SHA256_FILE Return a lowercase SHA-256 digest using MATLAB's Java runtime.
message_digest = java.security.MessageDigest.getInstance('SHA-256');
input_stream = java.io.FileInputStream(java.io.File(file_path));
digest_stream = java.security.DigestInputStream(input_stream, message_digest);
cleanup = onCleanup(@() digest_stream.close());
while digest_stream.read() ~= -1
end
clear cleanup;
hash_bytes = typecast(message_digest.digest(), 'uint8');
digest = lower(reshape(dec2hex(hash_bytes, 2).', 1, []));
end
