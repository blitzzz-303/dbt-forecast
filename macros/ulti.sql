{% macro run_predict(source_tbl, _ts_fld, _cat_fld, _label_fld) -%}

    call prophet_predict('{"tbl_source": "{{source_tbl}}",
                        "tbl_target": "{{this}}",
                        "_ts_fld":"{{_ts_fld}}", "_cat_fld":"{{_cat_fld}}", "_label_fld": "{{_label_fld}}"}');

{%- endmacro %}
