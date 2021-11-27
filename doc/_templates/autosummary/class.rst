{{ fullname | escape | underline }}


.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}   
   {% if attributes %}
   .. rubric:: Attributes
   .. autosummary::

   {% for item in attributes %}
   {% if item not in inherited_members %}
      ~{{name}}.{{item}}
   {%- endif %}
   {%- endfor %}
   {%- endif %}
   
   .. rubric:: Methods
   .. autosummary::

   {% for item in methods %}
   {% if item not in inherited_members %}
      ~{{name}}.{{item}}
   {%- endif %}
   {%- endfor %}

   ..
      Methods

{% block methods %}
{% for item in methods %}
{% if item not in inherited_members %}
   .. automethod:: {{ item }}
{%- endif %}
{%- endfor %}
{%- endblock %}