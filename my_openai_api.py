# encoding: utf-8
import json
import time
import uuid
from threading import Thread

import torch
from flask import Flask, current_app, request, Blueprint, stream_with_context
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
from transformers.generation.streamers import TextIteratorStreamer
from marshmallow import validate
from flasgger import APISpec, Schema, Swagger, fields
from apispec.ext.marshmallow import MarshmallowPlugin
from apispec_webframeworks.flask import FlaskPlugin


class Transformers():
    def __init__(self, app=None, tokenizer=None, model=None):
        if app is not None:
            self.init_app(app, tokenizer, model)

    def init_app(self, app, tokenizer=None, model=None):
        self.tokenizer = tokenizer
        self.model = model
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)


tfs = Transformers()

models_bp = Blueprint('Models', __name__, url_prefix='/v1/models')
chat_bp = Blueprint('Chat', __name__, url_prefix='/v1/chat')


def sse(line, field="data"):
    return "{}: {}\n\n".format(
        field, json.dumps(line, ensure_ascii=False) if isinstance(line, dict) else line)


def empty_cache():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def create_app():
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(models_bp)
    app.register_blueprint(chat_bp)

    @app.after_request
    def after_request(resp):
        empty_cache()
        return resp

    # Init Swagger
    spec = APISpec(
        title='My OpenAI api',
        version='0.0.1',
        openapi_version='3.0.2',
        plugins=[
            FlaskPlugin(),
            MarshmallowPlugin(),
        ],
    )

    bearer_scheme = {"type": "http", "scheme": "bearer"}
    spec.components.security_scheme("bearer", bearer_scheme)
    template = spec.to_flasgger(
        app,
        paths=[list_models, create_chat_completion]
    )

    app.config['SWAGGER'] = {"openapi": "3.0.2"}
    Swagger(app, template=template)

    # Init transformers
    model_name = "./Yi-34B-Chat-4bits"
    tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained(model_name)

    tfs.init_app(app, tokenizer, model)

    return app


class ModelSchema(Schema):
    id = fields.Str()
    object = fields.Str(dump_default="model", metadata={"example": "model"})
    created = fields.Int(dump_default=lambda: int(time.time()), metadata={"example": 1695402567})
    owned_by = fields.Str(dump_default="owner", metadata={"example": "owner"})


class ModelListSchema(Schema):
    object = fields.Str(dump_default="list", metadata={"example": "list"})
    data = fields.List(fields.Nested(ModelSchema), dump_default=[])


class ChatMessageSchema(Schema):
    role = fields.Str(required=True, metadata={"example": "system"})
    content = fields.Str(required=True, metadata={"example": "You are a helpful assistant."})


class CreateChatCompletionSchema(Schema):
    model = fields.Str(required=True, metadata={"example": "gpt-3.5-turbo"})
    messages = fields.List(
        fields.Nested(ChatMessageSchema), required=True,
        metadata={"example": [
            ChatMessageSchema().dump({"role": "system", "content": "You are a helpful assistant."}),
            ChatMessageSchema().dump({"role": "user", "content": "Hello!"})
        ]}
    )
    temperature = fields.Float(load_default=1.0, metadata={"example": 1.0})
    top_p = fields.Float(load_default=1.0, metadata={"example": 1.0})
    n = fields.Int(load_default=1, metadata={"example": 1})
    max_tokens = fields.Int(load_default=1000, metadata={"example": 1000})
    stream = fields.Bool(load_default=False, example=False)
    presence_penalty = fields.Float(load_default=0.0, example=0.0)
    frequency_penalty = fields.Float(load_default=0.0, example=0.0)


class ChatCompletionChoiceSchema(Schema):
    index = fields.Int(metadata={"example": 0})
    message = fields.Nested(ChatMessageSchema, metadata={
        "example": ChatMessageSchema().dump(
                {"role": "assistant", "content": "\n\nHello there, how may I assist you today?"}
        )})
    finish_reason = fields.Str(
        validate=validate.OneOf(["stop", "length", "content_filter", "function_call"]),
        metadata={"example": "stop"})


class ChatCompletionSchema(Schema):
    id = fields.Str(
            dump_default=lambda: uuid.uuid4().hex,
            metadata={"example": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7"})
    object = fields.Constant("chat.completion")
    created = fields.Int(dump_default=lambda: int(time.time()), metadata={"example": 1695402567})
    model = fields.Str(metadata={"example": "gpt-3.5-turbo"})
    choices = fields.List(fields.Nested(ChatCompletionChoiceSchema))


class ChatDeltaSchema(Schema):
    role = fields.Str(metadata={"example": "assistant"})
    content = fields.Str(required=True, metadata={"example": "Hello"})


class ChatCompletionChunkChoiceSchema(Schema):
    index = fields.Int(metadata={"example": 0})
    delta = fields.Nested(ChatDeltaSchema, metadata={"example": ChatDeltaSchema().dump(
        {"role": "assistant", "example": "Hello"})})
    finish_reason = fields.Str(
        validate=validate.OneOf(["stop", "length", "content_filter", "function_call"]),
        metadata={"example": "stop"})


class ChatCompletionChunkShema(Schema):
    id = fields.Str(
            dump_default=lambda: uuid.uuid4().hex,
            metadata={"example": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7"})
    object = fields.Constant("chat.completion.chunk")
    created = fields.Int(dump_default=lambda: int(time.time()), metadata={"example": 1695402567})
    model = fields.Str(metadata={"example": "gpt-3.5-turbo"})
    choices = fields.List(fields.Nested(ChatCompletionChunkChoiceSchema))

@models_bp.route("")
def list_models():
    """
    List models
    ---
    get:
      tags:
        - Models
      description: Lists the currently available models, \
and provides basic information about each one such as the owner and availability.
      security:
        - bearer: []
      responses:
        200:
          description: Models returned
          content:
            application/json:
              schema: ModelListSchema
    """

    model = ModelSchema().dump({"id": "gpt-3.5-turbo"})
    return ModelListSchema().dump({"data": [model]})


@stream_with_context
def stream_chat_generate(messages,temperature=1.0,top_p=0.8,):
    delta = ChatDeltaSchema().dump(
            {"role": "assistant"})
    choice = ChatCompletionChunkChoiceSchema().dump(
            {"index": 0, "delta": delta, "finish_reason": None})

    yield sse(
        ChatCompletionChunkShema().dump({
            "model": "gpt-3.5-turbo",
            "choices": [choice]})
    )

    input_ids = tfs.tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True,
                                          return_tensors='pt')
    generation_kwargs = dict(input_ids=input_ids.to('cuda'), streamer=tfs.streamer, temperature=temperature, top_p=top_p)
    thread = Thread(target=tfs.model.generate, kwargs=generation_kwargs)
    thread.start()
    for content in tfs.streamer:
        if not content:
            continue
        content = content.replace('<|im_end|>', '\n')
        empty_cache()
        delta = ChatDeltaSchema().dump(
                {"content": content})
        choice = ChatCompletionChunkChoiceSchema().dump(
                {"index": 0, "delta": delta, "finish_reason": None})

        yield sse(
            ChatCompletionChunkShema().dump({
                "model": "gpt-3.5-turbo",
                "choices": [choice]})
        )

    choice = ChatCompletionChunkChoiceSchema().dump(
            {"index": 0, "delta": {}, "finish_reason": "stop"})

    yield sse(
        ChatCompletionChunkShema().dump({
            "model": "gpt-3.5-turbo",
            "choices": [choice]})
    )

    yield sse('[DONE]')


@chat_bp.route("/completions", methods=['POST'])
def create_chat_completion():
    """Create chat completion
    ---
    post:
      tags:
        - Chat
      description: Creates a model response for the given chat conversation.
      requestBody:
        request: True
        content:
          application/json:
            schema: CreateChatCompletionSchema
      security:
        - bearer: []
      responses:
        200:
          description: ChatCompletion return
          content:
            application/json:
              schema:
                oneOf:
                  - ChatCompletionSchema
                  - ChatCompletionChunkShema
    """

    create_chat_completion = CreateChatCompletionSchema().load(request.json)

    if create_chat_completion["stream"]:
        return current_app.response_class(
            stream_chat_generate(create_chat_completion["messages"]),
            mimetype="text/event-stream"
        )
    else:
        input_ids = tfs.tokenizer.apply_chat_template(conversation=create_chat_completion["messages"], tokenize=True, add_generation_prompt=True,
                                          return_tensors='pt')
        output_ids = tfs.model.generate(input_ids.to('cuda'))
        response = tfs.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        message = ChatMessageSchema().dump(
                {"role": "assistant", "content": response})
        choice = ChatCompletionChoiceSchema().dump(
                {"index": 0, "message": message, "finish_reason": "stop"})
        return ChatCompletionSchema().dump({
            "model": "gpt-3.5-turbo",
            "choices": [choice]})

app = create_app()

if __name__ == '__main__':
    try:
        import ngrok
        import logging

        logging.basicConfig(level=logging.INFO)
        listener = ngrok.werkzeug_develop()
    except Exception:
        pass

    app.run(debug=False, host="0.0.0.0", port=8081)
